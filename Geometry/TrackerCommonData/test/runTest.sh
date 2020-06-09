#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${LOCAL_TEST_DIR}/python/testDDCutTubsFromPointsAlgorithm.py
F2=${LOCAL_TEST_DIR}/python/testDDPixBarLayerUpgradeAlgorithm.py
F3=${LOCAL_TEST_DIR}/python/testDDPixFwdDiskAlgo.py
F4=${LOCAL_TEST_DIR}/python/testDDPixPhase1FwdDiskAlgorithm.py
F5=${LOCAL_TEST_DIR}/python/testDDTIDAxialCableAlgorithm.py

echo " testing Geometry/TrackerCommonData"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
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
