#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_triggerresults_test_file_cfg.py || die 'Failure using create_triggerresults_test_file_cfg.py' $?

file=testTriggerResults.root

cmsRun ${LOCAL_TEST_DIR}/test_readTriggerResults_cfg.py "$file" || die "Failure using test_readTriggerResults_cfg.py $file" $?

# The old files read below were generated as follows.
#
#     Check out the release indicated in the filename. Then cherry pick
#     the commit that adds the original version of the file
#     DataFormats/Common/test/TestWriteTriggerResults.cc.
#     Except for BuildFile.xml, this only adds new test files.
#     There may be minor conflicts or issues in test/BuildFile.xml
#     that need to be resolved.
#
# Run cmsRun with DataFormats/Common/test/create_triggerresults_test_file_cfg.py
# as the configuration and rename the file that creates.

# Note that TriggerResults is declared in the classes_def.xml file with
# a requirement that the product is always written with split level 0.
# Most other raw data products are written with the default split level
# for the output file. That is why all the test input files in this
# shell script were written with split level 0.

oldFiles="testTriggerResults_CMSSW_13_0_0_split_0.root  testTriggerResults_CMSSW_13_1_0_pre3_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/Common/data/$file) || die "Failure edmFileInPath DataFormats/Common/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_readTriggerResults_cfg.py "$inputfile" || die "Failed to read old file $file" $?
done

exit 0
