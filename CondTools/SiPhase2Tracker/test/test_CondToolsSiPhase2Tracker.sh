 #!/bin/bash -ex
function die { echo $1: status $2 ; exit $2; }
TEST_DIR=$CMSSW_BASE/src/CondTools/SiPhase2Tracker/test
echo "test dir: $TEST_DIR"

printf "testing writing Phase2 Outer Tracker Lorentz Angle \n\n"
## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/SiPhase2OuterTrackerLorentzAngleWriter_cfg.py || die "Failure running SiPhase2OuterTrackerLorentzAngleWriter_cfg.py " $?
cmsRun ${TEST_DIR}/SiPhase2OuterTrackerLorentzAngleReader_cfg.py || die "Failure running SiPhase2OuterTrackerLorentzAngleReader_cfg.py " $?

printf "testing writing Phase2 Outer Tracker Bad Strips \n\n"
## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/SiPhase2BadStripChannelBuilder_cfg.py algorithm=1 || die "Failure running SiPhase2BadStripChannelBuilder_cfg.py (naive)" $?
cmsRun ${TEST_DIR}/SiPhase2BadStripChannelBuilder_cfg.py algorithm=1 || die "Failure running SiPhase2BadStripChannelBuilder_cfg.py (random)" $?
cmsRun ${TEST_DIR}/SiPhase2BadStripChannelReader_cfg.py  || die "Failure running SiPhase2BadStripChannelReader_cfg.py" $?
cmsRun ${TEST_DIR}/SiPhase2BadStripChannelReader_cfg.py fromESSource=True || die "Failure running SiPhase2BadStripChannelReader_cfg.py fromESSource=True" $?

printf "testing writing Phase2 Tracker Cabling Map (test) \n\n"
## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_write.py || die "Failure running DTCCablingMapTestProducer_write.py" $?
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_retrieve.py || die "Failure running DTCCablingMapTestProducer_retrieve.py" $?
cmsRun ${TEST_DIR}/DTCCablingMapTestProducer_dump.py || die "Failure running DTCCablingMapTestProducer_dump.py" $?

printf "testing writing Phase2 Tracker Cabling Map  \n\n"
## need to be in order (don't read before writing)
cmsRun ${TEST_DIR}/DTCCablingMapProducer_write.py || die "Failure running DTCCablingMapProducer_write.py" $?
cmsRun ${TEST_DIR}/DTCCablingMapProducer_retrieve.py || die "Failure running DTCCablingMapProducer_retrieve.py " $?
cmsRun ${TEST_DIR}/DTCCablingMapProducer_dump.py || die "Failure running DTCCablingMapProducer_dump.py" $?
