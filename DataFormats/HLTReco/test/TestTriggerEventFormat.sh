#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_TriggerEvent_test_file_cfg.py || die 'Failure using create_TriggerEvent_test_file_cfg.py' $?

file=testTriggerEvent.root

cmsRun ${LOCAL_TEST_DIR}/test_readTriggerEvent_cfg.py "$file" || die "Failure using test_readTriggerEvent_cfg.py $file" $?

oldFiles="testTriggerEvent_CMSSW_13_0_0.root testTriggerEvent_CMSSW_13_1_0_pre3.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/HLTReco/data/$file) || die "Failure edmFileInPath DataFormats/HLTReco/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_readTriggerEvent_cfg.py "$inputfile" || die "Failed to read old file $file" $?
done

exit 0
