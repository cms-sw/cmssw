#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=$CMSSW_BASE/src/FWCore/Framework/test

(cmsRun ${TEST_DIR}/test_1_thread_es_prefetch_cfg.py 2>&1) | grep 'Maximum concurrent running modules: 1' || die "grep failed to find 'Maximum concurrent running modules: 1'" $?
