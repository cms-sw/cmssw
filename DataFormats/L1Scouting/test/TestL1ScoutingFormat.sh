#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_L1Scouting_test_file_cfg.py || die 'Failure using create_L1Scouting_test_file_cfg.py' $?

file=testL1Scouting.root

cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py --inputFile "$file" || die "Failure using read_L1Scouting_cfg.py $file" $?

# The old files read below were generated as follows.
#
# Check out the release in the filename and use it without modification to make
# files with split level 99 (maximum possible splitting for each product)
#
# Then execute:
#
#   cmsRun DataFormats/L1Scouting/test/create_L1Scouting_test_file_cfg.py
#
# Rename the output file.
#
# The versions of the classes are encoded in the filenames in
# alphabetical order. This order is also the order the classes
# appear in classes_def.xml.
#
# For the split level 0 files, do the exact same thing except
# add the following to the output module configuration.
#     "splitLevel = cms.untracked.int32(0)"

# test file for muon, jet, e/gamma and energy sums data formats
oldFiles="testL1Scouting_v3_v3_v3_v3_v3_14_0_0_split_99.root testL1Scouting_v3_v3_v3_v3_v3_14_0_0_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$file) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py --inputFile "$inputfile" --bmtfStubVersion 0 || die "Failed to read old file $file" $?
done

# added BMTF input stubs data format
oldFiles="testL1Scouting_v3_v3_v3_v3_v3_v3_14_1_0_pre5_split_99.root testL1Scouting_v3_v3_v3_v3_v3_v3_14_1_0_pre5_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$file) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py --inputFile "$inputfile" --bmtfStubVersion 3 || die "Failed to read old file $file" $?
done

# added Calo tower data format
oldFiles="testL1Scouting_v3_v3_v3_v3_v3_v3_15_0_1_split_99.root testL1Scouting_v3_v3_v3_v3_v3_v3_15_0_1_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$file) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py --inputFile "$inputfile" --bmtfStubVersion 3 --caloTowerVersion 3 || die "Failed to read old file $file" $?
done

exit 0
