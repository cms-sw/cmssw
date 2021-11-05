#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo " testing CondTools/SiStrip"

for entry in "${LOCAL_TEST_DIR}/"SiStrip*Builder_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done
