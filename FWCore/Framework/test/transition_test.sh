#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 0 ) || die 'Failure running cmsRun transition_test_cfg.py 0' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 1 ) || die 'Failure running cmsRun transition_test_cfg.py 1' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 2 ) || die 'Failure running cmsRun transition_test_cfg.py 2' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 3 ) || die 'Failure running cmsRun transition_test_cfg.py 3' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 4 ) || die 'Failure running cmsRun transition_test_cfg.py 4' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 5 ) || die 'Failure running cmsRun transition_test_cfg.py 5' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 6 ) || die 'Failure running cmsRun transition_test_cfg.py 6' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 7 ) || die 'Failure running cmsRun transition_test_cfg.py 7' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 8 ) || die 'Failure running cmsRun transition_test_cfg.py 8' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 9 ) || die 'Failure running cmsRun transition_test_cfg.py 9' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 10 ) || die 'Failure running cmsRun transition_test_cfg.py 10' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 11 ) || die 'Failure running cmsRun transition_test_cfg.py 11' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 12 ) || die 'Failure running cmsRun transition_test_cfg.py 12' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 13 ) || die 'Failure running cmsRun transition_test_cfg.py 13' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 14 ) || die 'Failure running cmsRun transition_test_cfg.py 14' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 15 ) || die 'Failure running cmsRun transition_test_cfg.py 15' $?
(cmsRun ${LOCAL_TEST_DIR}/transition_test_cfg.py 16 ) || die 'Failure running cmsRun transition_test_cfg.py 16' $?
