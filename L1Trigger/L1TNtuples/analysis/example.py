#!/usr/bin/env python

"""
Example L1TNtuple analysis program
"""


import ROOT


# apparently not needed...
# ROOT.gSystem.Load("libL1TriggerL1TNtuples")

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(1)
ROOT.TH1.SetDefaultSumw2(True)
ROOT.gStyle.SetOptStat(0)


def eventLoop(filename):
    f = ROOT.TFile(filename)

    treeL1   = f.Get("l1NtupleProducer/L1Tree")
    treeTow  = f.Get("l1CaloTowerTreeProducer/L1CaloTowerTree")
#    tree_l1ex = f.Get("l1ExtraTreeProducer/L1ExtraTree")
    treeL1up = f.Get("l1UpgradeTreeProducer/L1UpgradeTree")

    treeJet  = f.Get("l1JetRecoTreeProducer/JetRecoTree")
#    treeEG   = f.Get("l1EGRecoTreeProducer/EGRecoTree")
#    treeTau  = f.Get("l1TauRecoTreeProducer/TauRecoTree")
#    treeMu   = f.Get("l1MuonRecoTreeProducer/MuonRecoTree")

    treeL1.AddFriend(treeTow)
#    treeL1.AddFriend(treeL1ex)
    treeL1.AddFriend(treeL1up)
    treeL1.AddFriend(treeJet)


    for jentry, event in enumerate(tree):
        if jentry >= nevents:
            break

        event      = treeL1.Event
        eventSim   = treeL1.Simulation
        eventRCT   = treeL1.RCT
        eventGCT   = treeL1.GCT
        eventDTTF  = treeL1.DTTF
        eventCSCTF = treeL1.CSCTF
        eventGMT   = treeL1.GMT
        eventGT    = treeL1.GT

        eventCaloTP = treeTow.CaloTP
        eventTower  = treeTow.L1CaloTower

        eventL1Up  = treeL1up.L1Upgrade

        eventJet   = treeJet.Jet

#        eventEG    = treeEG.EG

#        eventTau   = treeTau.Tau

#        eventMuon  = treeMu.Mu


    # Print hists to file
    c = ROOT.TCanvas()

    
    h.Draw("")
    h.SetTitleOffset(0.55, 'Z')
    c.Print("plot.pdf")

    f.Close()  # make sure I go last! otherwise your hists will be NoneType


if __name__ == "__main__":
    towercorr()
