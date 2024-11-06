#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_GlobalObjectMapRecord_test_file_cfg.py || die 'Failure using create_GlobalObjectMapRecord_test_file_cfg.py' $?

file=testGlobalObjectMapRecord.root

cmsRun ${LOCAL_TEST_DIR}/test_readGlobalObjectMapRecord_cfg.py "$file" || die "Failure using test_readGlobalObjectMapRecord_cfg.py $file" $?

# The old files read below were generated as follows.
#
#     Check out the release in the filename and cherry pick the commit that
#     adds the original version of the file:
#     DataFormats/L1TGlobal/test/TestWriteGlobalObjectMapRecord.cc
#
# Run cmsRun with DataFormats/L1TGlobal/test/create_GlobalObjectMapRecord_test_file_cfg.py
# as the configuration and rename the file that creates.
#
# By default, split level 99 is used (maximum possible splitting).
# If the suffix "_split_0" is near the end of the filename, the
# following was added to the configuration of the output module:
#     "splitLevel = cms.untracked.int32(0)"

oldFiles="testGlobalObjectMapRecord_CMSSW_13_0_0_split_99.root testGlobalObjectMapRecord_CMSSW_13_0_0_split_0.root"
oldFiles+=" testGlobalObjectMapRecord_CMSSW_13_1_0_pre3_split_99.root testGlobalObjectMapRecord_CMSSW_13_1_0_pre3_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1TGlobal/data/$file) || die "Failure edmFileInPath DataFormats/L1TGlobal/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_readGlobalObjectMapRecord_cfg.py "$inputfile" || die "Failed to read old file $file" $?
done

exit 0
