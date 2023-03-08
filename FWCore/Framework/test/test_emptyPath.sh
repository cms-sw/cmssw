#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
F1=${LOCAL_TEST_DIR}/test_emptyPathWithTask_cfg.py
F2=${LOCAL_TEST_DIR}/test_emptyEndPathWithTask_cfg.py

(cmsRun $F1 ) || die "Failure using $F1" $?
(cmsRun $F2 ) || die "Failure using $F2" $?


