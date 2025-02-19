import uproot
import awkward as ak
import numpy as np

# Define the NanoAOD file and TTree
#file_path = "step2_inNANOAOD.root"  # Replace with your NanoAOD file path
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
        print(t_idx, vertices, event.tracksters_raw_energy[t_idx])
        print(t_idx, vertex_multiplicity, event.tracksters_raw_energy[t_idx])
