#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_cfg.py ) || die 'Failure using test_esproducerlooper_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_stop_cfg.py ) || die 'Failure using test_esproducerlooper_stop_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_override_cfg.py ) || die 'Failure using test_esproducerlooper_override_cfg.py' $?
#(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_cfg.py ) || die 'Failure using test_esproducerlooper_prefer_cfg.py' $?
#(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_not_source_cfg.py ) || die 'Failure using test_esproducerlooper_prefer_not_source_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_producer_cfg.py ) || die 'Failure using test_esproducerlooper_prefer_producer_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_module_change_looper_cfg.py ) || die 'Failure using test_module_change_looper_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_edlooper_consumes_cfg.py) || die 'Failure using test_edlooper_consumes_cfg.py' $?
