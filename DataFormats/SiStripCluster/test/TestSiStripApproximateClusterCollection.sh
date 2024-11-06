#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_SiStripApproximateClusterCollection_test_file_cfg.py || die 'Failure using create_SiStripApproximateClusterCollection_test_file_cfg.py' $?

file=testSiStripApproximateClusterCollection.root

cmsRun ${LOCAL_TEST_DIR}/test_readSiStripApproximateClusterCollection_cfg.py "$file" || die "Failure using test_readSiStripApproximateClusterCollection_cfg.py $file" $?

# The old files read below were generated as follows.
#
#     Check out the release in the filename and cherry pick the commit that
#     adds the original version of the file:
#     DataFormats/SiStripCluster/test/TestWriteSiStripApproximateClusterCollection.cc
#
# Run cmsRun with DataFormats/SiStripCluster/test/create_SiStripApproximateClusterCollection_test_file_cfg.py 
# as the configuration and rename the file that creates.
#
# By default, split level 99 is used (maximum possible splitting).
# If the suffix "_split_0" is near the end of the filename, the
# following was added to the configuration of the output module:
#     "splitLevel = cms.untracked.int32(0)"

oldFiles="testSiStripApproximateClusterCollection_CMSSW_13_2_4_split_99.root testSiStripApproximateClusterCollection_CMSSW_13_2_4_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/SiStripCluster/data/$file) || die "Failure edmFileInPath DataFormats/SiStripCluster/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_readSiStripApproximateClusterCollection_cfg.py "$inputfile" || die "Failed to read old file $file" $?
done

exit 0
