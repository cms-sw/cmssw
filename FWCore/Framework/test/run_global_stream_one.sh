#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/test_global_modules_cfg.py
F2=${LOCAL_TEST_DIR}/test_stream_modules_cfg.py
F3=${LOCAL_TEST_DIR}/test_one_modules_cfg.py
(cmsRun $F1 ) || die "Failure using $F1" $?
(cmsRun $F2 ) || die "Failure using $F2" $?
(cmsRun $F3 ) || die "Failure using $F3" $?

