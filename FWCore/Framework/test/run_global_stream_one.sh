#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ; echo === Log file === ; cat ${3:-/dev/null} ; echo === End log file === ; exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"

F1=${LOCAL_TEST_DIR}/test_global_modules_cfg.py
F2=${LOCAL_TEST_DIR}/test_stream_modules_cfg.py
F3=${LOCAL_TEST_DIR}/test_one_modules_cfg.py
F4=${LOCAL_TEST_DIR}/test_limited_modules_cfg.py
(cmsRun $F1 ) >& log_test_global_modules || die "Failure using $F1" $? log_test_global_modules
# These greps are testing callWhenNewProductsRegistered
grep "global::StreamIntProducer TriggerResults" log_test_global_modules > /dev/null || die "grep failed to find 'global::StreamIntProducer TriggerResults'" $? log_test_global_modules
grep "global::StreamIntAnalyzer TriggerResults" log_test_global_modules > /dev/null || die "grep failed to find 'global::StreamIntAnalyzer TriggerResults'" $? log_test_global_modules
(cmsRun $F2 ) >& log_test_stream_modules || die "Failure using $F2" $? log_test_stream_modules
grep "stream::GlobalIntAnalyzer TriggerResults" log_test_stream_modules > /dev/null || die "grep failed to find 'stream::GlobalIntAnalyzer TriggerResults'" $? log_test_stream_modules
(cmsRun $F3 ) >& log_test_one_modules || die "Failure using $F3" $? log_test_one_modules
grep "one::SharedResourcesAnalyzer TriggerResults" log_test_one_modules > /dev/null || die "grep failed to find 'one::SharedResourcesAnalyzer TriggerResults'" $? log_test_one_modules
(cmsRun $F4 ) >& log_test_limited_modules || die "Failure using $F4" $? log_test_limited_modules
grep "limited::StreamIntAnalyzer TriggerResults" log_test_limited_modules > /dev/null || die "grep failed to find 'limited::StreamIntAnalyzer TriggerResults'" $? log_test_limited_modules

echo cmsRun FWCore/Framework/test/testRunLumiCaches_cfg.py
cmsRun ${LOCAL_TEST_DIR}/testRunLumiCaches_cfg.py >& testRunLumiCaches_cfg.log || die "Failure using testRunLumiCaches_cfg.py" $? testRunLumiCaches_cfg.log

#the last few lines of the output are the printout from the
# ConcurrentModuleTimer service detailing how much time was
# spent in 2,3 or 4 modules running simultaneously.
touch empty_file

(cmsRun ${LOCAL_TEST_DIR}/test_no_concurrent_module_cfg.py ) >& log_test_no_concurrent_module
cat log_test_no_concurrent_module | tail -n 3 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file || die "Failure using test_no_concurrent_module_cfg.py" $? log_test_no_concurrent_module

(cmsRun ${LOCAL_TEST_DIR}/test_limited_concurrent_module_cfg.py ) >& log_test_limited_concurrent_module
cat log_test_limited_concurrent_module | tail -n 3 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file || die "Failure using test_limited_concurrent_module_cfg.py" $? log_test_limited_concurrent_module

echo cmsRun modules_2_concurrent_lumis_cfg.py
(cmsRun ${LOCAL_TEST_DIR}/modules_2_concurrent_lumis_cfg.py ) &> log_modules_2_concurrent_lumis || die "cmsRun modules_2_concurrent_lumis_cfg.py" $? log_modules_2_concurrent_lumis
