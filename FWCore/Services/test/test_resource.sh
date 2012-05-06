#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/test_resource_succeed_cfg.py
F2=${LOCAL_TEST_DIR}/test_resource_rss_fail_cfg.py
F3=${LOCAL_TEST_DIR}/test_resource_time_fail_cfg.py
F4=${LOCAL_TEST_DIR}/test_resource_vsize_fail_cfg.py

(cmsRun $F1 ) || die "Failure using $F1" $?
!(cmsRun $F2 ) || die "Failure using $F2" $?
!(cmsRun $F3 ) || die "Failure using $F3" $?
!(cmsRun $F4 ) || die "Failure using $F4" $?
