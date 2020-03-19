#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

F1=${LOCAL_TEST_DIR}/test_global_modules_cfg.py
F2=${LOCAL_TEST_DIR}/test_stream_modules_cfg.py
F3=${LOCAL_TEST_DIR}/test_one_modules_cfg.py
F4=${LOCAL_TEST_DIR}/test_limited_modules_cfg.py
(cmsRun $F1 ) >& log_test_global_modules || die "Failure using $F1" $?
# These greps are testing callWhenNewProductsRegistered
grep "global::StreamIntProducer TriggerResults" log_test_global_modules > /dev/null || die "grep failed to find 'global::StreamIntProducer TriggerResults'" $?
grep "global::StreamIntAnalyzer TriggerResults" log_test_global_modules > /dev/null || die "grep failed to find 'global::StreamIntAnalyzer TriggerResults'" $?
(cmsRun $F2 ) >& log_test_stream_modules || die "Failure using $F2" $?
grep "stream::GlobalIntAnalyzer TriggerResults" log_test_stream_modules > /dev/null || die "grep failed to find 'stream::GlobalIntAnalyzer TriggerResults'" $?
(cmsRun $F3 ) >& log_test_one_modules || die "Failure using $F3" $?
grep "one::SharedResourcesAnalyzer TriggerResults" log_test_one_modules > /dev/null || die "grep failed to find 'one::SharedResourcesAnalyzer TriggerResults'" $?
(cmsRun $F4 ) >& log_test_limited_modules || die "Failure using $F4" $?
grep "limited::StreamIntAnalyzer TriggerResults" log_test_limited_modules > /dev/null || die "grep failed to find 'limited::StreamIntAnalyzer TriggerResults'" $?

#the last few lines of the output are the printout from the
# ConcurrentModuleTimer service detailing how much time was
# spent in 2,3 or 4 modules running simultaneously.
touch empty_file

(cmsRun ${LOCAL_TEST_DIR}/test_no_concurrent_module_cfg.py 2>&1) | tail -n 3 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file || die "Failure using test_no_concurrent_module_cfg.py" $?

(cmsRun ${LOCAL_TEST_DIR}/test_limited_concurrent_module_cfg.py 2>&1) | tail -n 3 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file || die "Failure using test_limited_concurrent_module_cfg.py" $?

echo cmsRun modules_2_concurrent_lumis_cfg.py
cmsRun ${LOCAL_TEST_DIR}/modules_2_concurrent_lumis_cfg.py || die "cmsRun modules_2_concurrent_lumis_cfg.py" $?

popd
