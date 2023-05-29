#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=$CMSSW_RELEASE_BASE/src/CalibPPS/TimingCalibration/test

if [ -e $CMSSW_BASE/src/CalibPPS/TimingCalibration/test ] ; then
   TEST_DIR=$CMSSW_BASE/src/CalibPPS/TimingCalibration/test
fi

cmsRun  $TEST_DIR/DiamondCalibrationWorker_cfg.py || die "HPTDC PCL failed at worker stage" $?
echo "HPTDC PCL worker succeeded"
cmsRun  $TEST_DIR/DiamondCalibrationHarvester_cfg.py || die "HPTDC PCL failed at harvester stage" $?
echo "HPTDC PCL harvester succeeded"
