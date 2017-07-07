#!/bin/sh

function die { echo $1: status $2; exit $2; }

cmsRun ${LOCAL_TEST_DIR}/ContentTest_cfg.py || die 'failed running cmsRun ContentTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/printeventsetupcontent_cfg.py || die 'failed running cmsRun printeventsetupcontent_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/geteventsetupcontent_cfg.py || die 'failed running cmsRun geteventsetupcontent_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/checkcacheidentifier_cfg.py || die 'failed running cmsRun checkcacheidentifier_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/emptysource_multiprocess_cfg.py || die 'failed running cmsRun emptysource_multiprocess_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/multiprocess_failedChild_cfg.py && die 'cmsRun multiprocess_failedChild_cfg.py did not fail as it should' $?
cmsRun ${LOCAL_TEST_DIR}/multiprocess_failedChild_and_continue_cfg.py || 'failed running multiprocess_failedChild_and_continue_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/multiprocess_failedChild_exception_cfg.py && die 'cmsRun multiprocess_failedChild_exception_cfg.py did not fail as it should' $?
cmsRun ${LOCAL_TEST_DIR}/multiprocess_failedChild_exception_and_continue_cfg.py || 'failed running multiprocess_failedChild_exception_and_continue_cfg.py' $?

