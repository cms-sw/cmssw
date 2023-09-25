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

fnames = ["file:/eos/cms/store/group/cat/datasets/NANOAODSIM/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/7B930101-EB91-4F4E-9B90-0861460DBD94.root"]

p = PostProcessor(outputDir=".",
                  inputFiles=fnames,
                  cut="Jet_pt>150",
                  modules=[exampleModuleConstr()],
                  provenance=True,
                  maxEntries=50000, #just read the first maxEntries events
                  )
p.run()
