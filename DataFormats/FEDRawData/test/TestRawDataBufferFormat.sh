#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_RawDataBuffer_test_file_cfg.py || die 'Failure using create_RawDataBuffer_test_file_cfg.py' $?

file=testRawDataBuffer.root

cmsRun ${LOCAL_TEST_DIR}/test_readRawDataBuffer_cfg.py "$file" || die "Failure using test_readRawDataBuffer_cfg.py $file" $?

exit 0
