import time
from pathlib import Path

from rich import print as rprint
import gemmi
import numpy as np
import pytest
import pandas as pd
import json

from pandda_gemmi.site_model import ResiduePainting, HeirarchicalSiteModelAlignedSequences, Site, get_sites
from pandda_gemmi.event_model.event import Event
from pandda_gemmi.fs import PanDDAInput
from pandda_gemmi.dataset import XRayDataset
from pandda_gemmi.serialize import output_residue_assignments, read_residue_assignments, output_msa, serialize_msa, unserialize_msa, output_msa, read_msa

@pytest.mark.skip()
def test_HeirarchicalSiteModelAlignedSequences():
    data_dirs = Path('data/XX01ZVNS2B')
    existing_events = {}
    existing_sites = {}

    # Get the datasets
    fs = PanDDAInput(data_dirs, data_dirs, 'dimple.pdb', 'dimple.mtz')
    datasets = {
        dataset_dir.dtag: XRayDataset.from_paths(
            dataset_dir.input_pdb_file,
            dataset_dir.input_mtz_file,
            dataset_dir.input_ligands,
            name=dataset_dir.dtag
        )
        for dataset_dir
        in fs.dataset_dirs.values()
    }

    # Make synthetic events
    event_data = {
        'XX01ZVNS2B-x10175': [12.5, 16.0, 10.5],
        'XX01ZVNS2B-x11441': [12.0, 16.0, 12.0],
        'XX01ZVNS2B-x12469': [14.5, 15.0, 12.5],
        'XX01ZVNS2B-x10225': [12.5, 16.0, 10.0],
        'XX01ZVNS2B-x10978': [-11.0, 11.0, 19.5],
        'XX01ZVNS2B-x10754': [-9.5, 6.5, 17.5],
        'XX01ZVNS2B-x10645': [20.5, 27.5, 24.5],
    }
    pandda_events = {
        (dtag, 1): Event(
            np.array([pos,]),
            None,
            0,
            np.array(pos),
        )
        for dtag, pos 
        in event_data.items()
    }

    # Get ref
    ref_dataset = datasets[
            min(
                datasets,
                key=lambda _dtag: datasets[_dtag].reflections.resolution()
            )
        ]

    # Get the sites
    sites = get_sites(
        datasets,
        pandda_events,
        ref_dataset,
        HeirarchicalSiteModelAlignedSequences(
            t=0.3, 
            debug=True,
            distance=10.0
            ),
        existing_events,
        existing_sites
    )

    rprint(sites)


@pytest.mark.skip()
def test_HeirarchicalSiteModelAlignedSequences_existing_sites():
    data_dirs = Path('data/XX01ZVNS2B')

    # Get the datasets
    fs = PanDDAInput(data_dirs, data_dirs, 'dimple.pdb', 'dimple.mtz')
    datasets = {
        dataset_dir.dtag: XRayDataset.from_paths(
            dataset_dir.input_pdb_file,
            dataset_dir.input_mtz_file,
            dataset_dir.input_ligands,
            name=dataset_dir.dtag
        )
        for dataset_dir
        in fs.dataset_dirs.values()
    }

    # Make synthetic events
    existing_event_data = {
        'XX01ZVNS2B-x10175': [12.5, 16.0, 10.5],
        'XX01ZVNS2B-x10978': [-11.0, 11.0, 19.5],
    }
    event_data = {
        'XX01ZVNS2B-x11441': [12.0, 16.0, 12.0],
        'XX01ZVNS2B-x12469': [14.5, 15.0, 12.5],
        'XX01ZVNS2B-x10225': [12.5, 16.0, 10.0],
        'XX01ZVNS2B-x10754': [-9.5, 6.5, 17.5],
        'XX01ZVNS2B-x10645': [20.5, 27.5, 24.5],
    }
    event_data.update(existing_event_data)
    pandda_events = {
        (dtag, 1): Event(
            np.array([pos,]),
            None,
            0,
            np.array(pos),
        )
        for dtag, pos 
        in event_data.items()
    }
    existing_pandda_events = {
        ('XX01ZVNS2B-x10175', 1): {'dtag': 'XX01ZVNS2B-x10175', 'event_idx': 1, 'site_idx': 1},
        ('XX01ZVNS2B-x10978', 1): {'dtag': 'XX01ZVNS2B-x10978', 'event_idx': 1, 'site_idx': 2},
    }
    existing_sites = {
        1: {'centroid': "", "Name": "", 'Comment': ...},
        2: {'centroid': "", "Name": "", 'Comment': ...},
    }

    # Get ref
    ref_dataset = datasets[
            min(
                datasets,
                key=lambda _dtag: datasets[_dtag].reflections.resolution()
            )
        ]

    # Get the sites
    sites = get_sites(
        datasets,
        pandda_events,
        ref_dataset,
        HeirarchicalSiteModelAlignedSequences(
            t=0.3, 
            debug=True,
            distance=10.0
            ),
        existing_pandda_events,
        existing_sites
    )

    rprint('Final sites:')
    rprint({site_number: site.event_ids for site_number, site in sites.items()})

def test_HeirarchicalSiteModelAlignedSequences_real_event_table():
    data_dirs = Path('data/XX01ZVNS2B')

    pandda_inspect_events_file = data_dirs / 'pandda_inspect_events.csv'
    pandda_inspect_events = pd.read_csv(pandda_inspect_events_file)

    # Make fake datasets by reusing the same dimple for each 
    base_path = data_dirs / 'XX01ZVNS2B-x3819'
    dataset_map = {
        'XX01ZVNS2B-x0827': {'pdb': data_dirs / 'XX01ZVNS2B-x0827' / 'dimple.pdb', 'mtz': base_path / 'dimple.mtz'},
        'XX01ZVNS2B-x0846': {'pdb': data_dirs / 'XX01ZVNS2B-x0846' / 'dimple.pdb', 'mtz': base_path / 'dimple.mtz'},
        'XX01ZVNS2B-x0182': {'pdb': data_dirs / 'XX01ZVNS2B-x0182' / 'dimple.pdb', 'mtz': base_path / 'dimple.mtz'},
    }
    for _dtag in pandda_inspect_events['dtag'].unique():
        if _dtag not in dataset_map:
            dataset_map[_dtag] = {'pdb': base_path / 'dimple.pdb', 'mtz': base_path / 'dimple.mtz'}

    datasets = {
        _dtag: XRayDataset.from_paths(
            _data['pdb'],
            _data['mtz'],
            None,
            name=_dtag
        )
        for _dtag, _data in dataset_map.items()
    }
    rprint(f'Got {len(datasets)} datasets')

    # Get ref
    ref_dataset = datasets[
            min(
                datasets,
                key=lambda _dtag: datasets[_dtag].reflections.resolution()
            )
        ]

    # SPlit events for two folds
    pandda_events_1 = {
        (_row['dtag'], _row['event_idx']): Event(
            np.array([_row['x'], _row['y'], _row['z']]),
            None,
            0,
            np.array([_row['x'], _row['y'], _row['z']]),
            score=_row['z_peak']
        )
        for _idx, _row
        in pandda_inspect_events.iloc[:int(len(pandda_inspect_events) / 2)].iterrows()
    }

    pandda_events_2 = {
        (_row['dtag'], _row['event_idx']): Event(
            np.array([_row['x'], _row['y'], _row['z']]),
            None,
            0,
            np.array([_row['x'], _row['y'], _row['z']]),
            score=_row['z_peak']
        )
        for _idx, _row
        in pandda_inspect_events.iloc[int(len(pandda_inspect_events) / 2):].iterrows()
    }

    # Define a manual site on the other half of the protein
    forced_sites = {
        1: Site(
            [],
            np.zeros(3),
            dtag='XX01ZVNS2B-x0846',
            residues=[('B', '70')]
        ),
    }

    # Get fold 1 events and sites
    sites_1, residue_allocations_1, msa_1 = get_sites(
        datasets,
        pandda_events_1,
        ref_dataset,
        ResiduePainting(
            t=0.3, 
            debug=True,
            distance=10.0
            ),
        None,
        None,
        forced_sites,
        None,
    )
    event_to_site = {
        _event_id: _site_idx
        for _site_idx
        in sites_1
        for _event_id
        in sites_1[_site_idx].event_ids
    }

    # Get fold 2 events and sites
    existing_pandda_events = {
        (_dtag, _event_idx): {'dtag': _dtag, 'event_idx': _event_idx, 'site_idx': event_to_site[(_dtag, _event_idx)]}
        for _dtag, _event_idx
        in pandda_events_1
    }

    rprint(f'# Sites')
    rprint(sites_1)
    rprint(f'# Residue Allocations')
    rprint(residue_allocations_1)
    rprint(f'# msa')
    # rprint(...)

    # output_msa(msa_1, None)
    assert unserialize_msa(serialize_msa(msa_1)) == msa_1 
    assert unserialize_msa(json.loads(json.dumps(serialize_msa(msa_1)))) == msa_1 
    path = 'test_msa.json'
    output_msa(msa_1, path) 
    new_msa = read_msa(path)
    assert msa_1 == new_msa
    print(f'Serialized and unserialized msa to json successfully!')


    path = 'test_residue_assignments.json'

    output_residue_assignments(residue_allocations_1, path)
    new_residue_assignments= read_residue_assignments(path)
    assert new_residue_assignments == residue_allocations_1
    print(f'Serialized and unserialized residue assignments to json successfully!')

    sites_2, residue_allocations_2, msa_2 = get_sites(
            datasets,
            pandda_events_2,
            ref_dataset,
            ResiduePainting(
                t=0.3, 
                debug=True,
                distance=10.0
                ),
            existing_pandda_events,
            sites_1,
            forced_sites,
            msa_1
        )    

    rprint('Final sites:')
    rprint({site_number: site.event_ids for site_number, site in sites_2.items()})


    rprint(f'###### Sites 2 ######')
    rprint(sites_2)
    rprint(f'###### Residue Allocations ######')
    rprint(residue_allocations_2)
    rprint(msa_2)