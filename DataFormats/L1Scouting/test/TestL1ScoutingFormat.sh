#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_L1Scouting_test_file_cfg.py || die 'Failure using create_L1Scouting_test_file_cfg.py' $?

file=testL1Scouting.root

cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py "$file" || die "Failure using read_L1Scouting_cfg.py $file" $?

oldFile="testL1Scouting_v3_v3_v3_v3_v3_13_3_0_pre5.root"
inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$oldFile) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$oldFile" $?
cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py "$inputfile" || die "Failed to read old file $oldFile" $?

exit 0