#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_L1Scouting_test_file_cfg.py || die 'Failure using create_L1Scouting_test_file_cfg.py' $?

file=testL1Scouting.root

cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py "$file" || die "Failure using read_L1Scouting_cfg.py $file" $?