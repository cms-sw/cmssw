#! /bin/bash

echo "Dumping GRun configuration"
# The FastSim configuration is managed through HLTrigger/Configuration/python/Tools/confdb.py,
# which is where the hltGetConfiguration command actually points to
hltGetConfiguration $(head -1 HLTVersionGRun) --fastsim > $CMSSW_BASE/src/FastSimulation/Configuration/python/HLT_GRun_cff.py
