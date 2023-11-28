#!/bin/sh

function die { echo $1: status $2; exit $2; }

echo ContentTest
cmsRun ${SCRAM_TEST_PATH}/ContentTest_cfg.py || die 'failed running cmsRun ContentTest_cfg.py' $?
echo printeventsetupcontent
cmsRun ${SCRAM_TEST_PATH}/printeventsetupcontent_cfg.py || die 'failed running cmsRun printeventsetupcontent_cfg.py' $?
echo geteventsetupcontent
cmsRun ${SCRAM_TEST_PATH}/geteventsetupcontent_cfg.py || die 'failed running cmsRun geteventsetupcontent_cfg.py' $?
echo checkcacheidentifier
cmsRun ${SCRAM_TEST_PATH}/checkcacheidentifier_cfg.py || die 'failed running cmsRun checkcacheidentifier_cfg.py' $?
echo testPathStatusFilter
cmsRun ${SCRAM_TEST_PATH}/testPathStatusFilter_cfg.py || die 'failed running cmsRun testPathStatusFilter_cfg.py' $?
echo sleepingModules
cmsRun ${SCRAM_TEST_PATH}/sleepingModules_cfg.py || die 'failed running cmsRun sleepingModules_cfg.py' $?
