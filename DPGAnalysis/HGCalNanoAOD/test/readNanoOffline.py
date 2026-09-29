import uproot
import awkward as ak
import numpy as np
import argparse
import sys

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--validation", action="store_true", help="Check Sim branches")
    args = parser.parse_args()

    file_path = args.filename
    validation = args.validation
    tree_name = "Events"

    # Open the ROOT file and load the TTree
    try:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            print("Available branches:")
            print(tree.keys())

            events = tree.arrays(library="ak")
            print("\nFields:", "\n\t".join(events[0].fields))
    except Exception as e:
        print(f"Error opening file or reading tree: {e}")
        sys.exit(1)

    # Event loop
    for i, event in enumerate(events):
        print(f"\n{'='*80}")
        print(f"Processing event {i}")
        print(f"{'='*80}")

        # LayerClusters
        print(f"\nFound {event.nHGCalLayerClusters} LayerClusters")
        for lc_idx in range(event.nHGCalLayerClusters):
            lcX = event.HGCalLayerClusters_position_x[lc_idx]
            lcY = event.HGCalLayerClusters_position_y[lc_idx]
            lcZ = event.HGCalLayerClusters_position_z[lc_idx]
            lcEta = event.HGCalLayerClusters_position_eta[lc_idx]
            lcPhi = event.HGCalLayerClusters_position_phi[lc_idx]
            lcE = event.HGCalLayerClusters_energy[lc_idx]
            print(f"  LC Idx {lc_idx}: pos=({lcX:.2f}, {lcY:.2f}, {lcZ:.2f}) eta-phi=({lcEta:.3f}, {lcPhi:.3f}) E={lcE:.2f}")

        # CLUE3D Tracksters
        print(f"\nFound {event.nticlTrackstersCLUE3DHigh} CLUE3D Tracksters")
        for t_idx in range(event.nticlTrackstersCLUE3DHigh):
            offset = event.ticlTrackstersCLUE3DHigh_oticlTrackstersCLUE3DHighvertices[t_idx]
            count = event.ticlTrackstersCLUE3DHigh_nticlTrackstersCLUE3DHighvertices[t_idx]
            vertices = event.ticlTrackstersCLUE3DHighvertices_vertices[offset : offset + count]
            vertex_multiplicity = event.ticlTrackstersCLUE3DHighvertices_vertex_mult[offset : offset + count]
            raw_energy = event.ticlTrackstersCLUE3DHigh_raw_energy[t_idx]
            print(f"  Trackster {t_idx}: vertices={len(vertices)}, raw_energy={raw_energy:.2f}")
            if len(vertices) > 0:
                print(f"    vertices: {list(zip(vertices, vertex_multiplicity))}")

        # TICLCandidates
        print(f"\nFound {event.nTICLCandidates} TICLCandidates")
        for cand_idx in range(event.nTICLCandidates):
            trackIdx = event.TICLCandidates_trackIdx[cand_idx]
            pt = event.TICLCandidates_pt[cand_idx]
            eta = event.TICLCandidates_eta[cand_idx]
            phi = event.TICLCandidates_phi[cand_idx]
            energy = event.TICLCandidates_energy[cand_idx]
            raw_energy = event.TICLCandidates_raw_energy[cand_idx]
            charge = event.TICLCandidates_charge[cand_idx]
            pdgID = event.TICLCandidates_pdgID[cand_idx]

            print(f"  TICLCandidate {cand_idx}:")
            print(f"    trackIdx: {trackIdx}")
            print(f"    pt={pt:.2f}, eta={eta:.3f}, phi={phi:.3f}")
            print(f"    energy={energy:.2f}, raw_energy={raw_energy:.2f}")
            print(f"    charge={charge}, pdgID={pdgID}")

            # Print linked tracksters with track boundary information
            try:
                # Calculate offset for this candidate in the TICLCandidatesExtra collection
                offset = sum(event.TICLCandidates_nTICLCandidatesExtra[:cand_idx])
                count = event.TICLCandidates_nTICLCandidatesExtra[cand_idx]
                if count > 0:
                    print(f"    Linked tracksters ({count}):")
                    for ts_idx in range(count):
                        idx = offset + ts_idx
                        trackster_idx = event.TICLCandidatesExtra_tracksterIndex[idx]
                        bX = event.TICLCandidatesExtra_track_boundaryX[idx]
                        bY = event.TICLCandidatesExtra_track_boundaryY[idx]
                        bZ = event.TICLCandidatesExtra_track_boundaryZ[idx]
                        bEta = event.TICLCandidatesExtra_track_boundaryEta[idx]
                        bPhi = event.TICLCandidatesExtra_track_boundaryPhi[idx]
                        bPx = event.TICLCandidatesExtra_track_boundaryPx[idx]
                        bPy = event.TICLCandidatesExtra_track_boundaryPy[idx]
                        bPz = event.TICLCandidatesExtra_track_boundaryPz[idx]

                        print(f"      Trackster {trackster_idx}:")
                        if bX != -999:
                            print(f"        Track boundary position: ({bX:.2f}, {bY:.2f}, {bZ:.2f})")
                            print(f"        Track boundary eta-phi: ({bEta:.3f}, {bPhi:.3f})")
                            print(f"        Track boundary momentum: ({bPx:.2f}, {bPy:.2f}, {bPz:.2f})")
                        else:
                            print(f"        Track boundary: No valid propagation")
            except (AttributeError, IndexError) as e:
                print(f"    Warning: Could not read track boundary info: {e}")

        # Sim2Reco connections for CLUE3D Tracksters
        print(f"\nExploring Sim2Reco connections for CLUE3D Tracksters")
        print(f"Found {event.nSimCP2ticlTrackstersCLUE3DHighByHits} SimCP connections")
        try:
            offset = 0
            for obj_idx in range(event.nSimCP2ticlTrackstersCLUE3DHighByHits):
                count = event.SimCP2ticlTrackstersCLUE3DHighByHits_nSimCP2ticlTrackstersCLUE3DHighByHitsLinks[obj_idx]
                elements = event.SimCP2ticlTrackstersCLUE3DHighByHitsLinks_index[offset : offset + count]
                scores = event.SimCP2ticlTrackstersCLUE3DHighByHitsLinks_score[offset : offset + count]
                sharedEnergy = event.SimCP2ticlTrackstersCLUE3DHighByHitsLinks_sharedEnergy[offset : offset + count]
                if len(elements) > 0:
                    print(f"  SimCP {obj_idx} -> Tracksters: {elements}, scores: {scores}, sharedE: {sharedEnergy}")
                offset += count
        except AttributeError as e:
            print(f"  AttributeError: {e}")

        # Validation mode: SimTICLCandidates
        if validation:
            print(f"\nFound {event.nSimTICLCandidates} SimTICLCandidates")
            for sim_idx in range(event.nSimTICLCandidates):
                trackIdx = event.SimTICLCandidates_trackIdx[sim_idx]
                pt = event.SimTICLCandidates_pt[sim_idx]
                eta = event.SimTICLCandidates_eta[sim_idx]
                phi = event.SimTICLCandidates_phi[sim_idx]
                energy = event.SimTICLCandidates_energy[sim_idx]
                raw_energy = event.SimTICLCandidates_raw_energy[sim_idx]
                charge = event.SimTICLCandidates_charge[sim_idx]
                pdgID = event.SimTICLCandidates_pdgID[sim_idx]

                print(f"  SimTICLCandidate {sim_idx}:")
                print(f"    trackIdx: {trackIdx}")
                print(f"    pt={pt:.2f}, eta={eta:.3f}, phi={phi:.3f}")
                print(f"    energy={energy:.2f}, raw_energy={raw_energy:.2f}")
                print(f"    charge={charge}, pdgID={pdgID}")

                # Print linked tracksters with track boundary information
                try:
                    # Calculate offset for this candidate in the SimTICLCandidatesExtra collection
                    offset = sum(event.SimTICLCandidates_nSimTICLCandidatesExtra[:sim_idx])
                    count = event.SimTICLCandidates_nSimTICLCandidatesExtra[sim_idx]
                    if count > 0:
                        print(f"    Linked tracksters ({count}):")
                        for ts_idx in range(count):
                            idx = offset + ts_idx
                            trackster_idx = event.SimTICLCandidatesExtra_tracksterIndex[idx]
                            bX = event.SimTICLCandidatesExtra_track_boundaryX[idx]
                            bY = event.SimTICLCandidatesExtra_track_boundaryY[idx]
                            bZ = event.SimTICLCandidatesExtra_track_boundaryZ[idx]
                            bEta = event.SimTICLCandidatesExtra_track_boundaryEta[idx]
                            bPhi = event.SimTICLCandidatesExtra_track_boundaryPhi[idx]
                            bPx = event.SimTICLCandidatesExtra_track_boundaryPx[idx]
                            bPy = event.SimTICLCandidatesExtra_track_boundaryPy[idx]
                            bPz = event.SimTICLCandidatesExtra_track_boundaryPz[idx]

                            print(f"      Trackster {trackster_idx}:")
                            if bX != -999:
                                print(f"        Track boundary position: ({bX:.2f}, {bY:.2f}, {bZ:.2f})")
                                print(f"        Track boundary eta-phi: ({bEta:.3f}, {bPhi:.3f})")
                                print(f"        Track boundary momentum: ({bPx:.2f}, {bPy:.2f}, {bPz:.2f})")
                            else:
                                print(f"        Track boundary: No valid propagation")
                except (AttributeError, IndexError) as e:
                    print(f"    Warning: Could not read track boundary info: {e}")

if __name__ == "__main__":
    main()
