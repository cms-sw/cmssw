#!/usr/bin/env python3
#
# Example of filling an histogram with nanoAODTools in a plain python script.
# 
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from importlib import import_module
import os
import sys
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


class ExampleAnalysis(Module):
    def __init__(self):
        self.writeHistFile = True

    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)

        self.h_vpt = ROOT.TH1F('sumpt', 'sumpt', 100, 0, 1000)
        self.addObject(self.h_vpt)

    def analyze(self, event):
        electrons = Collection(event, "Electron")
        muons = Collection(event, "Muon")
        jets = Collection(event, "Jet")
        eventSum = ROOT.TLorentzVector()

        # select events with at least 2 muons
        if len(muons) >= 2:
            for lep in muons:  # loop on muons
                eventSum += lep.p4()
            for lep in electrons:  # loop on electrons
                eventSum += lep.p4()
            for j in jets:  # loop on jets
                eventSum += j.p4()
            self.h_vpt.Fill(eventSum.Pt())  # fill histogram

        return True


preselection = "Jet_pt[0] > 250"
files = ["root://eoscms.cern.ch//eos/cms/store/user/cmsbuild/store/group/cat/datasets/NANOAODSIM/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/7B930101-EB91-4F4E-9B90-0861460DBD94.root"]

p = PostProcessor(".",
                  files,
                  cut=preselection,
                  branchsel=None,
                  modules=[ExampleAnalysis()],
                  noOut=True,
                  histFileName="histOut.root",
                  histDirName="plots",
                  )
p.run()
