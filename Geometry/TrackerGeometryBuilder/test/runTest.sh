#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo " testing Geometry/TrackerGeomtryBuilder"

for entry in "${LOCAL_TEST_DIR}/python"/test*
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done
