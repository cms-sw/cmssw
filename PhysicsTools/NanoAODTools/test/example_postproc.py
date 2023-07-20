#!/usr/bin/env python3
#
# Example of running the postprocessor to skim events with a cut, and 
# adding a new variable using a Module.
#
from PhysicsTools.NanoAODTools.postprocessing.examples.exampleModule import *

from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from importlib import import_module
import os
import sys
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

fnames = ["root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL17NanoAODv9/DYJetsToLL_M-500to700_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/30000/B0F32031-B096-254E-9E65-68206AD35B5F.root"]

p = PostProcessor(outputDir=".",
                  inputFiles=fnames,
                  cut="Jet_pt>150",
                  modules=[exampleModuleConstr()],
                  provenance=True,
                  maxEntries=5000, #just read the first maxEntries events
                  )
p.run()
