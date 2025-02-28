##########################################################
#
# The file trackingNtuple.root has the data I need.
# I can section it off into jets.
# I then put these jets into a new n-tuple, 
#
##########################################################

import matplotlib.pyplot as plt
import ROOT
from ROOT import TFile
from myjets import getLists, createJets, matchArr
import numpy as np

# Possible pT cut
pTCut = 100

# Load existing tree
file = TFile("trackingNtuple.root")
old_tree = file["trackingNtuple"]["tree"]

# Create a new ROOT file to store the new TTree
new_file = ROOT.TFile("new_tree.root", "RECREATE")

# Create a new subdirectory in the new file
new_directory = new_file.mkdir("trackingNtuple")

# Change the current directory to the new subdirectory
new_directory.cd()

# Create a new TTree with the same structure as the old one but empty
new_tree = old_tree.CloneTree(0)  

# Create a variable to hold the new leaves' data (a list of floats)
new_leaf_etadiffs = ROOT.std.vector('float')()
new_leaf_phidiffs = ROOT.std.vector('float')()
new_leaf_rjet = ROOT.std.vector('float')()
new_leaf_jet_eta = ROOT.std.vector('float')()
new_leaf_jet_phi = ROOT.std.vector('float')()
new_leaf_jet_pt = ROOT.std.vector('float')()


# Create a new branch in the tree
new_tree.Branch("sim_etadiffs", new_leaf_etadiffs)
new_tree.Branch("sim_phidiffs", new_leaf_phidiffs)
new_tree.Branch("sim_rjet", new_leaf_rjet)
new_tree.Branch("sim_jet_eta", new_leaf_jet_eta)
new_tree.Branch("sim_jet_phi", new_leaf_jet_phi)
new_tree.Branch("sim_jet_pt", new_leaf_jet_pt)

# Loop over entries in the old tree
for i in range(old_tree.GetEntries()):
    old_tree.GetEntry(i)

    # Clear the vector to start fresh for this entry
    new_leaf_etadiffs.clear()
    new_leaf_phidiffs.clear()
    new_leaf_rjet.clear()
    new_leaf_jet_eta.clear()
    new_leaf_jet_phi.clear()
    new_leaf_jet_pt.clear()

    # Creates the lists that will fill the leaves
    pTList, etaList, phiList, massList = getLists(old_tree, 0)
    jets = createJets(pTList, etaList, phiList, massList)
    
    jetIndex = np.array([])
    eta_diffs = np.array([])
    phi_diffs = np.array([])
    rjets = np.array([])
    jet_eta = np.array([])
    jet_phi = np.array([])
    jet_pt = np.array([])

    for jet in jets:
        const = jet.constituents_array()
        # Reorder particles within jet (jet clustering does not respect original index)
        jetIndex = np.append(jetIndex, matchArr(const["pT"], pTList)) # order restored by matching pT
        jetIndex = jetIndex.astype(int)

        # Compute the distance to jet
        etaval = const["eta"]-jet.eta
        phival = const["phi"]-jet.phi
        eta_diffs = np.append(eta_diffs, etaval)
        phi_diffs = np.append(phi_diffs, phival)

        rval = np.sqrt(etaval**2 + phival**2)
        rjets = np.append(rjets, rval)

        # Save values of closest jet
        jet_eta_val = np.ones(len(const["eta"]))*jet.eta
        jet_eta = np.append(jet_eta, jet_eta_val)
        jet_phi_val = np.ones(len(const["eta"]))*jet.phi
        jet_phi = np.append(jet_phi, jet_phi_val)
        jet_pt_val = np.ones(len(const["eta"]))*jet.pt
        jet_pt = np.append(jet_pt, jet_pt_val)
    
    # print(jet_pt)

    # Reorder branches appropriately
    length = len(eta_diffs)
    re_eta_diffs = np.zeros(length)
    re_phi_diffs = np.zeros(length)
    re_rjets = np.zeros(length)
    re_jet_eta = np.zeros(length)
    re_jet_phi = np.zeros(length)
    re_jet_pt = np.zeros(length)

    for i in range(length):
        re_eta_diffs[jetIndex[i]] = eta_diffs[i]
        re_phi_diffs[jetIndex[i]] = phi_diffs[i]
        re_rjets[jetIndex[i]] = rjets[i]
        re_jet_eta[jetIndex[i]] = jet_eta[i]
        re_jet_phi[jetIndex[i]] = jet_phi[i]
        re_jet_pt[jetIndex[i]] = jet_pt[i]

    # Add the list elements to the vector
    for value in re_eta_diffs:
        new_leaf_etadiffs.push_back(value)
    for value in re_phi_diffs:
        new_leaf_phidiffs.push_back(value)
    for value in re_rjets:
        new_leaf_rjet.push_back(value)
    for value in re_jet_eta:
        new_leaf_jet_eta.push_back(value)
    for value in re_jet_phi:
        new_leaf_jet_phi.push_back(value)
    for value in re_jet_pt:
        new_leaf_jet_pt.push_back(value)

    # Fill the tree with the new values
    new_tree.Fill()

    # break # Just for testing purposes

# new_tree.Scan("sim_pt@.size():sim_jet_pt@.size()")

# Write the tree back to the file
new_tree.Write()

# Debugging: print new_tree events
# new_tree.GetEntry(0) # only look at first event
# for i in range(3): # only look at first 3 tracks in event
#     print(f"Track {i}: sim_phi = {new_tree.sim_phi[i]}\t sim_eta = {new_tree.sim_eta[i]} \t sim_pt = {new_tree.sim_pt[i]} \t sim_rjet = {new_tree.sim_rjet[i]}") 

new_file.Close()
file.Close()

