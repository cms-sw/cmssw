#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=src/FWCore/Framework/test

(cmsRun ${TEST_DIR}/test_1_thread_es_mixed_prefetch_cfg.py 2>&1) | grep 'Maximum concurrent running modules: 1' || die "grep failed to find 'Maximum concurrent running modules: 1'" $?
(cmsRun ${TEST_DIR}/test_1_thread_es_no_prefetch_cfg.py 2>&1) | grep 'Maximum concurrent running modules: 1' || die "grep failed to find 'Maximum concurrent running modules: 1'" $?
(cmsRun ${TEST_DIR}/test_1_thread_es_prefetch_cfg.py 2>&1) | grep 'Maximum concurrent running modules: 1' || die "grep failed to find 'Maximum concurrent running modules: 1'" $?
