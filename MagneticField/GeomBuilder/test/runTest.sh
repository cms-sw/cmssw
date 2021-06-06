#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${LOCAL_TEST_DIR}/python/dumpMFGeometry.py
F2=${LOCAL_TEST_DIR}/python/testMFGeometry.py

echo " testing MagneticField/GeomBuilder"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"cmsRun dumpMFGeometry.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testMFGeometry.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
