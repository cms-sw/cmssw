#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper.cfg ) || die 'Failure using test_esproducerlooper.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper.cfg ) || die 'Failure using test_esproducerlooper_stop.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_override.cfg ) || die 'Failure using test_esproducerlooper_override.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer.cfg ) || die 'Failure using test_esproducerlooper_prefer.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_not_source.cfg ) || die 'Failure using test_esproducerlooper_prefer_not_source.cfg' $?
(cmsRun ${LOCAL_TEST_DIR}/test_esproducerlooper_prefer_producer.cfg ) || die 'Failure using test_esproducerlooper_prefer_producer.cfg' $?
