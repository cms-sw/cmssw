#!/bin/sh

function die { echo $1: status $2; exit $2; }

cmsRun ${LOCAL_TEST_DIR}/ContentTest_cfg.py || die 'failed running cmsRun ContentTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/printeventsetupcontent_cfg.py || die 'failed running cmsRun printeventsetupcontent_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/geteventsetupcontent_cfg.py || die 'failed running cmsRun geteventsetupcontent_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/checkcacheidentifier_cfg.py || die 'failed running cmsRun checkcacheidentifier_cfg.py' $?
