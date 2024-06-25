#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_L1Scouting_test_file_cfg.py || die 'Failure using create_L1Scouting_test_file_cfg.py' $?

file=testL1Scouting.root

cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py --inputFile "$file" || die "Failure using read_L1Scouting_cfg.py $file" $?

# test file for muon, jet, e/gamma and energy sums data formats
oldFile="testL1Scouting_v3_v3_v3_v3_v3_13_3_0_pre5.root"
inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$oldFile) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$oldFile" $?
cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py --inputFile "$inputfile" --bmtfStubVersion 0 || die "Failed to read old file $oldFile" $?

# added BMTF input stubs data format
oldFile="testL1Scouting_v3_v3_v3_v3_v3_v3_14_1_0_pre4.root"
inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$oldFile) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$oldFile" $?
cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py --inputFile "$inputfile" --bmtfStubVersion 3 || die "Failed to read old file $oldFile" $?

exit 0
