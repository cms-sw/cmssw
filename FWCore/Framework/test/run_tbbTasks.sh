#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"

F1=${LOCAL_TEST_DIR}/test_tbb_threads_cfg.py
F2="-n 8 ${LOCAL_TEST_DIR}/test_tbb_threads_from_commandline_cfg.py"
F3="--numThreads 8 ${LOCAL_TEST_DIR}/test_tbb_threads_from_commandline_cfg.py"
F4=${LOCAL_TEST_DIR}/test_tbb_default_threads_cfg.py

(cmsRun $F1 ) || die "Failure using cmsRun $F1" $?
(cmsRun $F2 ) || die "Failure using cmsRun $F2" $?
(cmsRun $F3 ) || die "Failure using cmsRun $F3" $?
(cmsRun $F4 ) || die "Failure using cmsRun $F4" $?

