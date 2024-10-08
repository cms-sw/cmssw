#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_SDSRawDataCollection_test_file_cfg.py || die 'Failure using create_SDSRawDataCollection_test_file_cfg.py' $?

file=testSDSRawDataCollection.root

cmsRun ${LOCAL_TEST_DIR}/read_SDSRawDataCollection_cfg.py "$file" || die "Failure using read_SDSRawDataCollection_cfg.py $file" $?

# The old files read below were generated as follows.
#
# Check out the release in the filename and use it without modification.
# Then execute:
#
#   cmsRun DataFormats/L1ScoutingRawData/test/create_SDSRawDataCollection_test_file_cfg.py
#
# Rename the output file. The versions of the class is encoded in the filename.
#
# Note that SDSRawDataCollection is declared in the classes_def.xml file with
# a requirement that the product is always written with split level 0.
# Most other raw data products are written with the default split level
# for the output file. That is why all the test input files in this
# shell script were written with split level 0.

oldFile="testSDSRawDataCollection_v3_CMSSW_14_0_0_split_0.root"
inputfile=$(edmFileInPath DataFormats/L1ScoutingRawData/data/$oldFile) || die "Failure edmFileInPath DataFormats/L1ScoutingRawData/data/$oldFile" $?
cmsRun ${LOCAL_TEST_DIR}/read_SDSRawDataCollection_cfg.py "$inputfile" || die "Failed to read old file $oldFile" $?

exit 0
