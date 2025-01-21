#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING CalibTracker/SiStripCommon ..."
echo cmsRun test_all_cfg.py
cmsRun ${SCRAM_TEST_PATH}/test_all_cfg.py || die "Failure running test_CalibTrackerSiStripCommon_cfg.py" $?

echo cmsRun testProduceCalibrationTree_cfg.py
cmsRun ${SCRAM_TEST_PATH}/testProduceCalibrationTree_cfg.py maxEvents=10 unitTest=True || die "Failure running produceCalibrationTree_template_cfg.py" $?
