#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_VectorDetId_test_file_cfg.py || die 'Failure using create_VectorDetId_test_file_cfg.py' $?

file=testVectorDetId.root

cmsRun ${LOCAL_TEST_DIR}/test_readVectorDetId_cfg.py "$file" || die "Failure using test_readVectorDetId_cfg.py $file" $?

# The old files read below were generated as follows.
#
#     Check out the 13_2_4 release and cherry pick the commit that
#     adds the original version of the file
#     DataFormats/DetId/test/TestWriteVectorDetId.cc.
#     Except for BuildFile.xml this only adds new test files.
#     There may be minor conflicts or issues in test/BuildFile.xml
#     that need to be resolved.
#
# Run cmsRun with DataFormats/DetId/test/create_VectorDetId_test_file_cfg.py
# as the configuration and rename the file that creates.
#
# By default, split level 99 is used (maximum possible splitting).
# If the suffix "_split_0" is near the end of the filename, the
# following was added to the configuration of the output module:
#     "splitLevel = cms.untracked.int32(0)"
#

oldFiles="testVectorDetId_CMSSW_13_2_4_split_99.root testVectorDetId_CMSSW_13_2_4_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/DetId/data/$file) || die "Failure edmFileInPath DataFormats/DetId/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_readVectorDetId_cfg.py "$inputfile" || die "Failed to read old file $file" $?
done

exit 0
