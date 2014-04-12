#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_cfg.py ) || die 'Failure using test_esproducerlooper_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_stop_cfg.py ) || die 'Failure using test_esproducerlooper_stop_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_override_cfg.py ) || die 'Failure using test_esproducerlooper_override_cfg.py' $?
#(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_cfg.py ) || die 'Failure using test_esproducerlooper_prefer_cfg.py' $?
#(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_not_source_cfg.py ) || die 'Failure using test_esproducerlooper_prefer_not_source_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_producer_cfg.py ) || die 'Failure using test_esproducerlooper_prefer_producer_cfg.py' $?
(cmsRun ${LOCAL_TEST_DIR}/test_module_change_looper_cfg.py ) || die 'Failure using test_module_change_looper_cfg.py' $?

