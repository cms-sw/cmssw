#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${SCRAM_TEST_PATH}/python/testDDCutTubsFromPointsAlgorithm.py
F2=${SCRAM_TEST_PATH}/python/testDDPixBarLayerUpgradeAlgorithm.py
F3=${SCRAM_TEST_PATH}/python/testDDPixFwdDiskAlgo.py
F4=${SCRAM_TEST_PATH}/python/testDDPixPhase1FwdDiskAlgorithm.py
F5=${SCRAM_TEST_PATH}/python/testDDTIDAxialCableAlgorithm.py

echo " testing Geometry/TrackerCommonData"

export tmpdir=${PWD}
echo "===== Test \"cmsRun testDDCutTubsFromPointsAlgorithm.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testDDPixBarLayerUpgradeAlgorithm.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
echo "===== Test \"cmsRun testDDPixFwdDiskAlgo.py\" ===="
(cmsRun $F3) || die "Failure using cmsRun $F4" $?
echo "===== Test \"cmsRun testDDPixPhase1FwdDiskAlgorithm.py\" ===="
(cmsRun $F4) || die "Failure using cmsRun $F5" $?
echo "===== Test \"cmsRun testDDTIDAxialCableAlgorithm.py\" ===="
(cmsRun $F5) || die "Failure using cmsRun $F23" $?
