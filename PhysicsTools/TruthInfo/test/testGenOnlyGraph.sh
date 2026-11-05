#!/bin/bash
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# A GEN-only job builds the truth graph without SIM, and a later job writes its metadata
# from the process history, also when that job has a module under the graph producer label.
set -ex
cmsRun ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraph_cfg.py
python3 ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraph_t.py genOnlyGraph.root
cmsRun ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraphMeta_cfg.py
python3 ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraph_t.py genOnlyGraphMeta.root
cmsRun ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraphMeta_cfg.py shadow
python3 ${LOCALTOP}/src/PhysicsTools/TruthInfo/test/genOnlyGraph_t.py genOnlyGraphMetaShadow.root
