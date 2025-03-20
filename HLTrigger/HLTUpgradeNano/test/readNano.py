import uproot
import awkward as ak
import numpy as np

# Define the NanoAOD file and TTree
# file_path = "step2_inNANOAOD.root"  # Replace with your NanoAOD file path
file_path = "step4.root"  # Replace with your NanoAOD file path
tree_name = "Events"

# Open the ROOT file and load the TTree
with uproot.open(file_path) as file:
    tree = file[tree_name]
    print(tree.keys())

    events = tree.arrays(library="ak")
    print("Fields:", "\n\t".join(events[0].fields))

# Event loop
for i, event in enumerate(events):
    print(f"Processing event {i}")
    print("Found {} tracksters".format(event.ntracksters))
    for t_idx, t in enumerate(range(event.ntracksters)):
        offset = event.tracksters_overtices[t_idx]
        count = event.tracksters_nvertices[t_idx]
        vertices = event.vertices_vertices[offset : offset + count]
        vertex_multiplicity = event.vertices_vertex_mult[offset : offset + count]
        print(
            t_idx,
            list(zip(vertices, vertex_multiplicity)),
            event.tracksters_raw_energy[t_idx],
        )
    print("Exploring connections,scores, and sharedEnergy")
    print("Connections for {} objects".format(event.nSimTS2TSMergeByHits))
    try:  # Offset pattern
        offset = 0
        entries = event.nSimTS2TSMergeByHits
        for obj_idx, obj in enumerate(range(entries - 1)):
            next_offset = event.SimTS2TSMergeByHits_oSimTS2TSMergeByHitsLinks[
                obj_idx + 1
            ]
            elements = event.SimTS2TSMergeByHitsLinks_index[offset:next_offset]
            scores = event.SimTS2TSMergeByHitsLinks_score[offset:next_offset]
            sharedEnergy = event.SimTS2TSMergeByHitsLinks_shardEnergy[
                offset:next_offset
            ]
            if len(elements) > 0:
                print("Offset ", obj_idx, elements, scores, sharedEnergy)
            offset = next_offset
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}")
    try:  # Count pattern
        offset = 0
        for obj_idx, obj in enumerate(range(event.nSimTS2TSMergeByHits)):
            count = event.SimTS2TSMergeByHits_nSimTS2TSMergeByHitsLinks[obj_idx]
            elements = event.SimTS2TSMergeByHitsLinks_index[offset : offset + count]
            scores = event.SimTS2TSMergeByHitsLinks_score[offset : offset + count]
            sharedEnergy = event.SimTS2TSMergeByHitsLinks_sharedEnergy[
                offset : offset + count
            ]
            if len(elements) > 0:
                print("Count ", obj_idx, elements, scores, sharedEnergy)
            offset = offset + count
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}")
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
        print(f"An AttributeError occurred: {e}")
