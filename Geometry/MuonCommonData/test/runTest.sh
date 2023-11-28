#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${SCRAM_TEST_PATH}/python/dumpMuonGeometry.py
F2=${SCRAM_TEST_PATH}/python/testMuonGeometry.py
F3=${SCRAM_TEST_PATH}/python/testMuonNumbering.py
F4=${SCRAM_TEST_PATH}/python/testDDGEMAngularAlgorithm.py

echo " testing Geometry/MuonCommonData"

export tmpdir=${PWD}
echo "===== Test \"cmsRun dumpMuonGeometry.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testMuonGeometry.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
echo "===== Test \"cmsRun testMuonNumbering.py\" ===="
(cmsRun $F3) || die "Failure using cmsRun $F3" $?
echo "===== Test \"cmsRun testDDGEMAngularAlgorithm.py\" ===="
(cmsRun $F4) || die "Failure using cmsRun $F4" $?
