#! /usr/bin/env python

import ROOT
from DataFormats.FWLite import Events, Handle
ROOT.gSystem.Load("libPhysicsToolsObjectSelectors")
reg = ROOT.objsel.Registry()
eventSelector = ROOT.objsel.WZEventSelector(reg)
eventSelector.setZbosonPoint();


files = ["/uscms_data/d2/malik/FWLITE_PAT_TUPLE_311/PatAnalyzerSkeletonSkim_ZMM311.root"]
events = Events (files)
handle  = Handle ("std::vector<pat::Muon>")

ROOT.gROOT.SetBatch()
ROOT.gROOT.SetStyle('Plain')
zmassHist = ROOT.TH1F ("zmass", "Z Candidate Mass", 50, 20, 220)

for event in events:
    reg.setEvent (event.object())
    if not eventSelector.passes( event.object() ):
        continue
    muons = eventSelector.tightMuonVec()
    numMuons = len (muons)
    if muons < 2: continue
    for outer in xrange (numMuons - 1):
        outerMuon = muons[outer]
        for inner in xrange (outer + 1, numMuons):
            innerMuon = muons[inner]
            if outerMuon.charge() * innerMuon.charge() >= 0:
                continue
            inner4v = ROOT.TLorentzVector (innerMuon.px(), innerMuon.py(),
                                           innerMuon.pz(), innerMuon.energy())
            outer4v = ROOT.TLorentzVector (outerMuon.px(), outerMuon.py(),
                                           outerMuon.pz(), outerMuon.energy())
            zmassHist.Fill( (inner4v + outer4v).M() )

print "done looping"
c1 = ROOT.TCanvas()
zmassHist.Draw()
c1.Print ("zmass_py.png")
print "done"    
