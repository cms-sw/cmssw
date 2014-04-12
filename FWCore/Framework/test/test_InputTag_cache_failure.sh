#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/test_InputTag_cache_failure_cfg.py
(cmsRun $F1 ) || die "Failure using $F1" $?
