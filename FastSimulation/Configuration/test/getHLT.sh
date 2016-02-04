#! /bin/bash

HLT=$(head -1 HLTVersionGRun)
echo "Dumping HLT configuration: $HLT"

# The FastSim configuration is managed through HLTrigger/Configuration/python/Tools/confdb.py,
# which is where the hltGetConfiguration command actually points to
hltGetConfiguration $HLT --fastsim > $CMSSW_BASE/src/FastSimulation/Configuration/python/HLT_GRun_cff.py
