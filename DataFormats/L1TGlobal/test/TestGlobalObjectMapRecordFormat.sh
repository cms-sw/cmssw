#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_GlobalObjectMapRecord_test_file_cfg.py || die 'Failure using create_GlobalObjectMapRecord_test_file_cfg.py' $?

file=testGlobalObjectMapRecord.root

cmsRun ${LOCAL_TEST_DIR}/test_readGlobalObjectMapRecord_cfg.py "$file" || die "Failure using test_readGlobalObjectMapRecord_cfg.py $file" $?

oldFiles="testGlobalObjectMapRecord_CMSSW_13_0_0.root testGlobalObjectMapRecord_CMSSW_13_1_0_pre3.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1TGlobal/data/$file) || die "Failure edmFileInPath DataFormats/L1TGlobal/data/$file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_readGlobalObjectMapRecord_cfg.py "$inputfile" || die "Failed to read old file $file" $?
done

exit 0
