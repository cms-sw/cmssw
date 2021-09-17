#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${LOCAL_TEST_DIR}/python/dumpMuonGeometry.py
F2=${LOCAL_TEST_DIR}/python/testMuonGeometry.py
F3=${LOCAL_TEST_DIR}/python/testMuonNumbering.py
F4=${LOCAL_TEST_DIR}/python/testDDGEMAngularAlgorithm.py

echo " testing Geometry/MuonCommonData"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"cmsRun dumpMuonGeometry.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testMuonGeometry.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
echo "===== Test \"cmsRun testMuonNumbering.py\" ===="
(cmsRun $F3) || die "Failure using cmsRun $F3" $?
echo "===== Test \"cmsRun testDDGEMAngularAlgorithm.py\" ===="
(cmsRun $F4) || die "Failure using cmsRun $F4" $?
