#!/bin/bash
set -eu

for PART in 1 2 3; do
  cmsRun "${SCRAM_TEST_PATH}/poolFileBlockTransition_cfg.py" --mode produce --part "$PART"
done
cmsRun -j transitionJobReport.xml "${SCRAM_TEST_PATH}/poolFileBlockTransition_cfg.py" --mode merge
cmsRun -j transitionNoCloneReport.xml "${SCRAM_TEST_PATH}/poolFileBlockTransition_cfg.py" --mode noclone
python3 "${SCRAM_TEST_PATH}/checkPoolFileBlockTransition.py" transitionJobReport.xml transitionNoCloneReport.xml
cmsRun "${SCRAM_TEST_PATH}/poolFileBlockTransition_cfg.py" --mode read
