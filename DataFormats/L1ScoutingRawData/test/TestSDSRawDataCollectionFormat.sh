#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_SDSRawDataCollection_test_file_cfg.py || die 'Failure using create_SDSRawDataCollection_test_file_cfg.py' $?

file=testSDSRawDataCollection.root

cmsRun ${LOCAL_TEST_DIR}/read_SDSRawDataCollection_cfg.py "$file" || die "Failure using read_SDSRawDataCollection_cfg.py $file" $?