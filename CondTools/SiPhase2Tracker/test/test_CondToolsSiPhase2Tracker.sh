 #!/bin/bash -ex
TEST_DIR=$CMSSW_BASE/src/CondTools/SiPhase2Tracker/test
echo "test dir: $TEST_DIR"

cmsRun ${TEST_DIR}/SiPhase2OuterTrackerLorentzAngleWriter_cfg.py

## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_write.py
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_retrieve.py
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_dump.py

## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/DTCCablingMapProducer_write.py
cmsRun ${TEST_DIR}/DTCCablingMapProducer_retrieve.py
cmsRun ${TEST_DIR}/DTCCablingMapProducer_dump.py
