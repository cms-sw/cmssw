#!/bin/sh

function die { echo $1: status $2; exit $2; }

pushd ${LOCAL_TMP_DIR}

echo ContentTest
cmsRun ${LOCAL_TEST_DIR}/ContentTest_cfg.py || die 'failed running cmsRun ContentTest_cfg.py' $?
echo printeventsetupcontent
cmsRun ${LOCAL_TEST_DIR}/printeventsetupcontent_cfg.py || die 'failed running cmsRun printeventsetupcontent_cfg.py' $?
echo geteventsetupcontent
cmsRun ${LOCAL_TEST_DIR}/geteventsetupcontent_cfg.py || die 'failed running cmsRun geteventsetupcontent_cfg.py' $?
echo checkcacheidentifier
cmsRun ${LOCAL_TEST_DIR}/checkcacheidentifier_cfg.py || die 'failed running cmsRun checkcacheidentifier_cfg.py' $?
echo testPathStatusFilter
cmsRun ${LOCAL_TEST_DIR}/testPathStatusFilter_cfg.py || die 'failed running cmsRun testPathStatusFilter_cfg.py' $?
echo sleepingModules
cmsRun ${LOCAL_TEST_DIR}/sleepingModules_cfg.py || die 'failed running cmsRun sleepingModules_cfg.py' $?

popd
