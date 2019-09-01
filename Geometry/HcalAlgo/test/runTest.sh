#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${LOCAL_TEST_DIR}/python/dumpHBGeom_cfg.py
F2=${LOCAL_TEST_DIR}/python/dumpHEPhase0Geom_cfg.py
F3=${LOCAL_TEST_DIR}/python/dumpHEPhase1Geom_cfg.py
F4=${LOCAL_TEST_DIR}/python/dumpHFGeom_cfg.py
F5=${LOCAL_TEST_DIR}/python/dumpHOGeom_cfg.py

echo " testing DetectorDescription/DDCMS"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"cmsRun dumpHBGeom_cfg.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun dumpHEPhase0Geom_cfg.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
echo "===== Test \"cmsRun dumpHEPhase1Geom_cfg.py\" ===="
(cmsRun $F3) || die "Failure using cmsRun $F3" $?
echo "===== Test \"cmsRun dumpHFGeom_cfg.py\" ===="
(cmsRun $F4) || die "Failure using cmsRun $F4" $?
echo "===== Test \"cmsRun dumpHOGeom_cfg.py\" ===="
(cmsRun $F5) || die "Failure using cmsRun $F5" $?
