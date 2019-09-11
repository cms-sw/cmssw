#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${LOCAL_TEST_DIR}/python/validateDTGeometry_cfg.py

echo " testing DetectorDescription/DDCMS"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"cmsRun validateDTGeometry_cfg.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
