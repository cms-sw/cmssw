#!/bin/bash
# Column check against a cmsDriver NANO configuration (generated offline)
set -e
cmsDriver.py nano --step NANO --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --era Run3_2026 \
  --conditions auto:phase1_2026_realistic --filein file:dummy.root --fileout file:out.root -n 1 --no_exec \
  --python_filename nano_cfg.py > driver.log 2>&1 || { tail -20 driver.log; exit 1; }
python3 ${LOCALTOP}/src/PhysicsTools/NanoAOD/test/patGenJet_check_columns.py nano_cfg.py
