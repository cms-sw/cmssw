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
    parser.add_argument("--ticlv", choices=["v4", "v5"], default="v5", help="TICL version")
    args = parser.parse_args()

    file_path = args.filename
    ticlVersion = args.ticlv
    validation = args.validation
    tree_name = "Events"

    # Open the ROOT file and load the TTree
    try:
        with uproot.open(file_path) as file:
            tree = file[tree_name]
            print(tree.keys())

            events = tree.arrays(library="ak")
            print("Fields:", "\n\t".join(events[0].fields))
    except Exception as e:
        print(f"Error opening file or reading tree: {e}")
        sys.exit(1)

    # Event loop
    for i, event in enumerate(events):
        print(f"Processing event {i}")
        #LayerClusters

        print("Found {} LayerClusters ".format(event.nhltMergeLayerClusters))
        for lc_idx in range(event.nhltMergeLayerClusters):
            lcX = event.hltMergeLayerClusters_position_x[lc_idx]
            lcY = event.hltMergeLayerClusters_position_y[lc_idx]
            lcZ = event.hltMergeLayerClusters_position_z[lc_idx]
            lcEta = event.hltMergeLayerClusters_position_eta[lc_idx]
            lcPhi = event.hltMergeLayerClusters_position_phi[lc_idx]
            lcE = event.hltMergeLayerClusters_energy[lc_idx]
            print(f"LC Idx {lc_idx} at ({lcX},{lcY},{lcZ}) (eta-phi) : ({lcEta}, {lcPhi}) with energy {lcE}")
        ## CLUE3D Tracksters ##
        print("Found {} CLUE3D Tracksters".format(event.nhltTiclTrackstersCLUE3DHigh))
        for t_idx in range(event.nhltTiclTrackstersCLUE3DHigh):
            offset = event.hltTiclTrackstersCLUE3DHigh_ohltTiclTrackstersCLUE3DHighvertices[t_idx]
            count = event.hltTiclTrackstersCLUE3DHigh_nhltTiclTrackstersCLUE3DHighvertices[t_idx]
            vertices = event.hltTiclTrackstersCLUE3DHighvertices_vertices[offset : offset + count]
            vertex_multiplicity = event.hltTiclTrackstersCLUE3DHighvertices_vertex_mult[offset : offset + count]
            print(
                t_idx,
                list(zip(vertices, vertex_multiplicity)),
                event.hltTiclTrackstersCLUE3DHigh_raw_energy[t_idx],
            )

        print("Exploring connections, scores, and sharedEnergy")
        print("Connections for {} objects".format(event.nSimCP2hltTiclTrackstersCLUE3DHighByHits))
        try:  # Offset pattern
            offset = 0
            for obj_idx in range(event.nSimCP2hltTiclTrackstersCLUE3DHighByHits- 1):
                next_offset = event.SimCP2hltTiclTrackstersCLUE3DHighByHits_oSimCP2hltTiclTrackstersCLUE3DHighByHitsLinks[
                    obj_idx + 1
                ]
                elements = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_index[offset:next_offset]
                scores = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_score[offset:next_offset]
                sharedEnergy = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_sharedEnergy[
                    offset:next_offset
                ]
                if len(elements) > 0:
                    print("Offset ", obj_idx, elements, scores, sharedEnergy)
                offset = next_offset
        except AttributeError as e:
            print(f"An AttributeError occurred (Offset): {e}")

        try:  # Count pattern
            offset = 0
            for obj_idx in range(event.nSimCP2hltTiclTrackstersCLUE3DHighByHits):
                count = event.SimCP2hltTiclTrackstersCLUE3DHighByHits_nSimCP2hltTiclTrackstersCLUE3DHighByHitsLinks[obj_idx]
                elements = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_index[offset : offset + count]
                scores = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_score[offset : offset + count]
                sharedEnergy = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_sharedEnergy[
                    offset : offset + count
                ]
                if len(elements) > 0:
                    print("Count ", obj_idx, elements, scores, sharedEnergy)
                offset += count
        except AttributeError as e:
            print(f"An AttributeError occurred (Count): {e}")

        ## TICLv5 Collections ##
        if(ticlVersion == "v5"):
            print("Found {} TICLCandidate tracksters".format(event.nhltTiclCandidate))
            try:  # Offset pattern
                offset = 0
                for obj_idx in range(event.nSimCP2hltTiclCandidateByHits - 1):
                    next_offset = event.SimCP2hltTiclCandidateByHits_oSimCP2hltTiclCandidateByHitsLinks[
                        obj_idx + 1
                    ]
                    elements = event.SimCP2hltTiclCandidateByHitsLinks_index[offset:next_offset]
                    scores = event.SimCP2hltTiclCandidateByHitsLinks_score[offset:next_offset]
                    sharedEnergy = event.SimCP2hltTiclCandidateByHitsLinks_sharedEnergy[
                        offset:next_offset
                    ]
                    if len(elements) > 0:
                        print("Offset ", obj_idx, elements, scores, sharedEnergy)
                    offset = next_offset
            except AttributeError as e:
                print(f"An AttributeError occurred (Offset): {e}")

            try:  # Count pattern
                offset = 0
                for obj_idx in range(event.nSimCP2hltTiclCandidateByHits):
                    count = event.SimSC2hltTiclCandidateByHits_nSimSC2hltTiclCandidateByHitsLinks[obj_idx]
                    elements = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_index[offset : offset + count]
                    scores = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_score[offset : offset + count]
                    sharedEnergy = event.SimCP2hltTiclTrackstersCLUE3DHighByHitsLinks_sharedEnergy[
                        offset : offset + count
                    ]
                    if len(elements) > 0:
                        print("Count ", obj_idx, elements, scores, sharedEnergy)
                    offset += count
            except AttributeError as e:
                print(f"An AttributeError occurred (Count): {e}")
    
            try:  # Count pattern
                offset = 0
                for obj_idx in range(event.nSimCP2hltTiclCandidateByHits):
                    count = event.SimCP2hltTiclCandidateByHits_nSimCP2hltTiclCandidateByHitsLinks[obj_idx]
                    elements = event.SimCP2hltTiclCandidateByHitsLinks_index[offset : offset + count]
                    scores = event.SimCP2hltTiclCandidateByHitsLinks_score[offset : offset + count]
                    sharedEnergy = event.SimCP2hltTiclCandidateByHitsLinks_sharedEnergy[
                        offset : offset + count
                    ]
                    if len(elements) > 0:
                        print("Count ", obj_idx, elements, scores, sharedEnergy)
                    offset += count
            except AttributeError as e:
                print(f"An AttributeError occurred (Count): {e}")


            #Reco2Sim
            
            try:  # Offset pattern
                offset = 0
                for obj_idx in range(event.nRecohltTiclCandidate2SimCPByHits - 1):
                    next_offset = event.RecohltTiclCandidate2SimCPByHits_oSimCP2hltTiclCandidateByHitsLinks[
                        obj_idx + 1
                    ]
                    elements = event.RecohltTiclCandidate2SimCPByHitsLinks_index[offset:next_offset]
                    scores = event.RecohltTiclCandidate2SimCPByHitsLinks_score[offset:next_offset]
                    sharedEnergy = event.RecohltTiclCandidate2SimCPByHitsLinks_sharedEnergy[
                        offset:next_offset
                    ]
                    if len(elements) > 0:
                        print("Offset ", obj_idx, elements, scores, sharedEnergy)
                    offset = next_offset
            except AttributeError as e:
                print(f"An AttributeError occurred (Offset): {e}")

            try:  # Count pattern
                offset = 0
                for obj_idx in range(event.nRecohltTiclCandidate2SimCPByHits):
                    count = event.RecohltTiclCandidate2SimCPByHits_nRecohltTiclCandidate2SimCPByHitsLinks[obj_idx]
                    elements = event.RecohltTiclCandidate2SimCPByHitsLinks_index[offset : offset + count]
                    scores = event.RecohltTiclCandidate2SimCPByHitsLinks_score[offset : offset + count]
                    sharedEnergy = event.RecohltTiclCandidate2SimCPByHitsLinks_sharedEnergy[offset : offset + count]
                    if len(elements) > 0:
                        print("Count ", obj_idx, elements, scores, sharedEnergy)
                    offset += count
            except AttributeError as e:
                print(f"An AttributeError occurred (Count): {e}")

            if(validation):
                print("Found {} simTICLCandidates".format(event.nhltSimTICLCandidates))
                for sim_idx in range(event.nhltSimTICLCandidates):
                    trackIdx = event.hltSimTICLCandidates_trackIdx[sim_idx]
                    if(trackIdx >= 0):
                        track_pt = event.hltGeneralTrack_pt[trackIdx]
                    else:
                        track_pt = np.nan;
                    print(
                        sim_idx,
                        trackIdx,
                        track_pt,
                        event.hltSimTICLCandidates_raw_energy[sim_idx],
                    )

        try:
            for i in range(event.nSimCl2CPWithFraction):
                print(
                    "SimCl {} is linked to CP {} with fraction {}".format(
                        i,
                        event.SimCl2CPWithFraction_index[i],
                        event.SimCl2CPWithFraction_fraction[i],
                    )
                )
        except AttributeError as e:
            print(f"An AttributeError occurred (SimCl2CPWithFraction): {e}")

if __name__ == "__main__":
    main()
