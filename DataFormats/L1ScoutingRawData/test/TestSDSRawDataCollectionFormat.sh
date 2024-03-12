#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_SDSRawDataCollection_test_file_cfg.py || die 'Failure using create_SDSRawDataCollection_test_file_cfg.py' $?

file=testSDSRawDataCollection.root

cmsRun ${LOCAL_TEST_DIR}/read_SDSRawDataCollection_cfg.py "$file" || die "Failure using read_SDSRawDataCollection_cfg.py $file" $?

oldFile="testSDSRawDataCollection_v3_CMSSW_13_3_0_pre5.root"
inputfile=$(edmFileInPath DataFormats/L1ScoutingRawData/data/$oldFile) || die "Failure edmFileInPath DataFormats/L1ScoutingRawData/data/$oldFile" $?
cmsRun ${LOCAL_TEST_DIR}/read_SDSRawDataCollection_cfg.py "$inputfile" || die "Failed to read old file $oldFile" $?

exit 0
