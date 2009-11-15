#! /usr/bin/env python

import ROOT
import sys
from DataFormats.FWLite import Events, Handle

# Make VarParsing object
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideAboutPythonConfigFile#VarParsing_Example
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('python')
options.parseArguments()

# Events takes either
# - single file name
# - list of file names
# - VarParsing options

# use Varparsing object
events = Events (options)

# create handle outside of loop
handle  = Handle ("std::vector<pat::Muon>")

# for now, label is just a tuple of strings that is initialized just
# like and edm::InputTag
label = ("selectedLayer1Muons")

# Create histograms, etc.
ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.gROOT.SetStyle('Plain') # white background
zmassHist = ROOT.TH1F ("zmass", "Z Candidate Mass", 50, 20, 220)

# loop over events
for event in events:
    # use getByLabel, just like in cmsRun
    event.getByLabel (label, handle)
    # get the product
    muons = handle.product()
    # use muons to make Z peak
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

# make a canvas, draw, and save it
c1 = ROOT.TCanvas()
zmassHist.Draw()
c1.Print ("zmass_py.png")
