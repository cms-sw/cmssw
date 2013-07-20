#! /usr/bin/env python

import ROOT
import sys
from DataFormats.FWLite import Events, Handle

files = ["patTuple.root"]
events = Events (files)
handle  = Handle ("std::vector<pat::Muon>")

# for now, label is just a tuple of strings that is initialized just
# like and edm::InputTag
label = ("cleanPatMuons")

f = ROOT.TFile("analyzerPython.root", "RECREATE")
f.cd()

muonPt  = ROOT.TH1F("muonPt", "pt",    100,  0.,300.)
muonEta = ROOT.TH1F("muonEta","eta",   100, -3.,  3.)
muonPhi = ROOT.TH1F("muonPhi","phi",   100, -5.,  5.) 

# loop over events
i = 0
for event in events:
    i = i + 1
    print  i
    # use getByLabel, just like in cmsRun
    event.getByLabel (label, handle)
    # get the product
    muons = handle.product()

    for muon in muons :
        muonPt.Fill( muon.pt() )
        muonEta.Fill( muon.eta() )
        muonPhi.Fill( muon.phi() )


f.cd()

muonPt.Write()
muonEta.Write()
muonPhi.Write()

f.Close()
