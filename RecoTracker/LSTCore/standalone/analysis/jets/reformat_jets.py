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

# Load existing tree
file =  TFile("/data2/segmentlinking/CMSSW_12_2_0_pre2/trackingNtuple_ttbar_PU200.root")
# file = TFile("trackingNtuple100.root")
old_tree = file["trackingNtuple"]["tree"]

# Create a new ROOT file to store the new TTree
new_file = ROOT.TFile("new_tree_temp.root", "RECREATE")

# Create a new subdirectory in the new file
new_directory = new_file.mkdir("trackingNtuple")

# Change the current directory to the new subdirectory
new_directory.cd()

# Create a new TTree with the same structure as the old one but empty
new_tree = old_tree.CloneTree(0)  

# Account for bug in 12_2_X branch
new_tree.SetBranchStatus("ph2_bbxi", False) 

# Create a variable to hold the new leaves' data (a list of floats)
new_leaf_deltaEta = ROOT.std.vector('float')()
new_leaf_deltaPhi = ROOT.std.vector('float')()
new_leaf_deltaR = ROOT.std.vector('float')()
new_leaf_jet_eta = ROOT.std.vector('float')()
new_leaf_jet_phi = ROOT.std.vector('float')()
new_leaf_jet_pt = ROOT.std.vector('float')()


# Create a new branch in the tree
new_tree.Branch("sim_deltaEta", new_leaf_deltaEta)
new_tree.Branch("sim_deltaPhi", new_leaf_deltaPhi)
new_tree.Branch("sim_deltaR", new_leaf_deltaR)
new_tree.Branch("sim_jet_eta", new_leaf_jet_eta)
new_tree.Branch("sim_jet_phi", new_leaf_jet_phi)
new_tree.Branch("sim_jet_pt", new_leaf_jet_pt)

# Loop over entries in the old tree
for ind in range(old_tree.GetEntries()):
    old_tree.GetEntry(ind)

    # Clear the vector to start fresh for this entry
    new_leaf_deltaEta.clear()
    new_leaf_deltaPhi.clear()
    new_leaf_deltaR.clear()
    new_leaf_jet_eta.clear()
    new_leaf_jet_phi.clear()
    new_leaf_jet_pt.clear()

    # Creates the lists that will fill the leaves
    pTList, etaList, phiList, massList = getLists(old_tree, hardSc=True, pTcut=True)
    cluster, jets = createJets(pTList, etaList, phiList, massList)
    
    jetIndex = np.array([])
    eta_diffs = np.array([])
    phi_diffs = np.array([])
    deltaRs = np.array([])
    jet_eta = np.array([])
    jet_phi = np.array([])
    jet_pt = np.array([])

    for j, jet in enumerate(jets):
        const = jet.constituents()
        c_len = len(const)
        c_pts = np.zeros(c_len)
        c_etas = np.zeros(c_len)
        c_phis = np.zeros(c_len)

        for k in range(c_len):
            c_pts[k] = const[k].pt()
            c_etas[k] = const[k].eta()
            c_phis[k] = const[k].phi()

        # Reorder particles within jet (jet clustering does not respect original index)
        jetIndex = np.append(jetIndex, matchArr(c_pts, c_etas, c_phis, 
                                                np.array(old_tree.sim_pt), np.array(old_tree.sim_eta), np.array(old_tree.sim_phi), 
                                                ind, j)) # order restored by matching pT
        jetIndex = jetIndex.astype(int)

        # Compute the distance to jet
        etaval = c_etas-jet.eta()
        phival = c_phis-jet.phi()
        eta_diffs = np.append(eta_diffs, etaval)
        phi_diffs = np.append(phi_diffs, phival)

        rval = np.sqrt(etaval**2 + np.arccos(np.cos(phival))**2)
        deltaRs = np.append(deltaRs, rval)

        # Save values of closest jet
        jet_eta_val = np.ones(c_len)*jet.eta()
        jet_eta = np.append(jet_eta, jet_eta_val)
        jet_phi_val = np.ones(c_len)*jet.phi()
        jet_phi = np.append(jet_phi, jet_phi_val)
        jet_pt_val = np.ones(c_len)*jet.pt()
        jet_pt = np.append(jet_pt, jet_pt_val)
    
    # Reordering branches appropriately
    length = len(np.array(old_tree.sim_pt))

    re_eta_diffs = np.ones(length)*(-999)
    re_phi_diffs = np.ones(length)*(-999)
    re_deltaRs = np.ones(length)*(-999)
    re_jet_eta = np.ones(length)*(-999)
    re_jet_phi = np.ones(length)*(-999)
    re_jet_pt = np.ones(length)*(-999)

    for i in range(len(jetIndex)):
        re_eta_diffs[jetIndex[i]] = eta_diffs[i]
        re_phi_diffs[jetIndex[i]] = phi_diffs[i]
        re_deltaRs[jetIndex[i]] = deltaRs[i]
        re_jet_eta[jetIndex[i]] = jet_eta[i]
        re_jet_phi[jetIndex[i]] = jet_phi[i]
        re_jet_pt[jetIndex[i]] = jet_pt[i]

    # Add the list elements to the vector
    for value in re_eta_diffs:
        new_leaf_deltaEta.push_back(value)
    for value in re_phi_diffs:
        new_leaf_deltaPhi.push_back(value)
    for value in re_deltaRs:
        new_leaf_deltaR.push_back(value)
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
#     print(f"Track {i}: sim_phi = {new_tree.sim_phi[i]}\t sim_eta = {new_tree.sim_eta[i]} \t sim_pt = {new_tree.sim_pt[i]} \t sim_deltaR = {new_tree.sim_deltaR[i]}") 

new_file.Close()
file.Close()

