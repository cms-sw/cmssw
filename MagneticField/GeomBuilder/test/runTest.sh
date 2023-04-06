#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${SCRAM_TEST_PATH}/python/dumpMFGeometry.py
F2=${SCRAM_TEST_PATH}/python/testMFGeometry.py

echo " testing MagneticField/GeomBuilder"

export tmpdir=${PWD}
echo "===== Test \"cmsRun dumpMFGeometry.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testMFGeometry.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
