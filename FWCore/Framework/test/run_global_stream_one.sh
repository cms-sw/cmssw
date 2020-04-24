#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/test_global_modules_cfg.py
F2=${LOCAL_TEST_DIR}/test_stream_modules_cfg.py
F3=${LOCAL_TEST_DIR}/test_one_modules_cfg.py
F4=${LOCAL_TEST_DIR}/test_limited_modules_cfg.py
(cmsRun $F1 ) || die "Failure using $F1" $?
(cmsRun $F2 ) || die "Failure using $F2" $?
(cmsRun $F3 ) || die "Failure using $F3" $?
(cmsRun $F4 ) || die "Failure using $F4" $?

#the last few lines of the output are the printout from the
# ConcurrentModuleTimer service detailing how much time was
# spent in 2,3 or 4 modules running simultaneously.
touch empty_file

(cmsRun ${LOCAL_TEST_DIR}/test_no_concurrent_module_cfg.py 2>&1) | tail -n 3 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file || die "Failure using test_no_concurrent_module_cfg.py" $?

(cmsRun ${LOCAL_TEST_DIR}/test_limited_concurrent_module_cfg.py 2>&1) | tail -n 3 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file || die "Failure using test_limited_concurrent_module_cfg.py" $?
