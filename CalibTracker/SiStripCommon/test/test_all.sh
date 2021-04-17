#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING CalibTracker/SiStripCommon ..."
cmsRun ${LOCAL_TEST_DIR}/test_all_cfg.py || die "Failure running test_CalibTrackerSiStripCommon_cfg.py" $? 
cmsRun ${LOCAL_TEST_DIR}/testProduceCalibrationTree_cfg.py maxEvents=10 unitTest=True || die "Failure running produceCalibrationTree_template_cfg.py" $?
