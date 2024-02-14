#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_VectorDetId_test_file_cfg.py || die 'Failure using create_VectorDetId_test_file_cfg.py' $?

file=testVectorDetId.root

cmsRun ${LOCAL_TEST_DIR}/test_readVectorDetId_cfg.py "$file" || die "Failure using test_readVectorDetId_cfg.py $file" $?

oldFiles="testVectorDetId_CMSSW_13_2_4.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/DetId/data/$file) || die "Failure edmFileInPath DataFormats/DetId/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_readVectorDetId_cfg.py "$inputfile" || die "Failed to read old file $file" $?
done

exit 0
