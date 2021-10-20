 #!/bin/bash -ex
TEST_DIR=$CMSSW_BASE/src/CondTools/SiPhase2Tracker/test
echo "test dir: $TEST_DIR"

printf "testing writing Phase2 Lorentz Angle \n\n"
## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/SiPhase2OuterTrackerLorentzAngleWriter_cfg.py
cmsRun ${TEST_DIR}/SiPhase2OuterTrackerLorentzAngleReader_cfg.py

printf "testing writing Phase2 Tracker Cabling Map (test) \n\n"
## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_write.py
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_retrieve.py
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_dump.py

printf "testing writing Phase2 Tracker Cabling Map  \n\n"
## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/DTCCablingMapProducer_write.py
cmsRun ${TEST_DIR}/DTCCablingMapProducer_retrieve.py
cmsRun ${TEST_DIR}/DTCCablingMapProducer_dump.py
