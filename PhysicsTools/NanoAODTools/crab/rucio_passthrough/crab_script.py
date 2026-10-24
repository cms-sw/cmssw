#!/usr/bin/env python3
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

# this takes care of converting the input files from CRAB
from PhysicsTools.NanoAODTools.postprocessing.utils.crabhelper import inputFiles, runsAndLumis

p = PostProcessor(".",
                  inputFiles(),
                  modules=[],
                  provenance=True,
                  fwkJobReport=True,
                  jsonInput=runsAndLumis())
p.run()

print("DONE")
