#!/bin/bash
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# A GEN-only job builds the truth graph without SIM.
set -ex
cmsRun ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraph_cfg.py
python3 ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraph_t.py genOnlyGraph.root
