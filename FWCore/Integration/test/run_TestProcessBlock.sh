#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo "testProcessBlock1"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock1_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlock1_cfg.py" $?

  # The MetaData ProcessBlock branch and the TTree should exist to hold the ProcessBlock
  # data. The Events branch should not exist because there were not any ProcessBlock branches
  # saved from an input file. Test that here:
  edmFileUtil -l -t MetaData -P file:testProcessBlock1.root > testProcessBlock1ContentsM.txt
  grep "Branch.* ProcessBlockHelper " testProcessBlock1ContentsM.txt || die "Check for existence of ProcessBlockHelper branch" $?
  grep "TTree.*ProcessBlocksPROD1" testProcessBlock1ContentsM.txt || die "Check for existence of ProcessBlocksPROD1 TTree" $?
  edmFileUtil -t Events -P file:testProcessBlock1.root > testProcessBlock1ContentsE.txt
  grep "Branch.* EventToProcessBlockIndexes " testProcessBlock1ContentsE.txt && die "Check for non-existence of eventToProcessBlockIndexes branch" 1

  echo "testProcessBlock2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlock2_cfg.py" $?

  echo "testProcessBlock3"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock3_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlock3_cfg.py" $?

  echo "testProcessBlock4"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock4_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlock4_cfg.py" $?

  echo "testProcessBlockMerge"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMerge_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockMerge_cfg.py" $?

  # The ProcessBlock Branches and TTrees should exist in this case. Test that here:
  edmFileUtil -l -t MetaData -P file:testProcessBlockMerge.root > testProcessBlockMContentsM.txt
  grep "Branch.* ProcessBlockHelper " testProcessBlockMContentsM.txt || die "Check for existence of ProcessBlockHelper branch" $?
  grep "TTree.*ProcessBlocksPROD1" testProcessBlockMContentsM.txt || die "Check for existence of ProcessBlocksPROD1 TTree" $?
  grep "TTree.*ProcessBlocksMERGE" testProcessBlockMContentsM.txt || die "Check for existence of ProcessBlocksMERGE TTree" $?
  edmFileUtil -t Events -P file:testProcessBlockMerge.root > testProcessBlockMContentsE.txt
  grep "Branch.* EventToProcessBlockIndexes " testProcessBlockMContentsE.txt || die "Check for existence of eventToProcessBlockIndexes branch" $?

  echo "testProcessBlockTEST"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockTEST_cfg.py || die "cmsRun testProcessBlockTEST_cfg.py" $?

  echo "testProcessBlockRead"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockRead_cfg.py &> testProcessBlockRead.log || die "cmsRun testProcessBlockRead_cfg.py" $?
  grep "InputProcessBlockIntAnalyzer::accessInputProcessBlock" testProcessBlockRead.log || die "Check that InputProcessBlockIntAnalyzer::accessInputProcessBlock was called" $?
  grep "InputProcessBlockIntFilter::accessInputProcessBlock" testProcessBlockRead.log || die "Check that InputProcessBlockIntFilter::accessInputProcessBlock was called" $?
  grep "InputProcessBlockIntProducer::accessInputProcessBlock" testProcessBlockRead.log || die "Check that InputProcessBlockIntProducer::accessInputProcessBlock was called" $?

  echo "testProcessBlock2Dropped"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock2Dropped_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlock2Dropped_cfg.py" $?

  # The ProcessBlock Branches and TTrees should not exist in this case because
  # all the ProcessBlock products are dropped. Test that here:
  edmFileUtil -l -t MetaData -P file:testProcessBlock2Dropped.root > testProcessBlock2DroppedContentsM.txt
  grep "Branch.* ProcessBlockHelper " testProcessBlock2DroppedContentsM.txt && die "Check for non-existence of ProcessBlockHelper branch" 1
  grep "TTree.*ProcessBlocksPROD1" testProcessBlock2DroppedContentsM.txt && die "Check for non-existence of ProcessBlocksPROD1 TTree" 1
  edmFileUtil -t Events -P file:testProcessBlock2Dropped.root > testProcessBlock2DroppedContentsE.txt
  grep "Branch.* EventToProcessBlockIndexes " testProcessBlock2DroppedContentsE.txt && die "Check for non-existence of eventToProcessBlockIndexes branch" 1

  # This one intentionally fails because the product content of the
  # files does not match (strict merging requirements for ProcessBlocks)
  echo "testProcessBlockFailMerge"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockFailMerge_cfg.py > /dev/null 2>&1 && die "cmsRun testProcessBlockFailMerge_cfg.py" 1

  echo "testProcessBlockMerge2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMerge2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockMerge2_cfg.py" $?

  echo "testProcessBlockMergeOfMergedFiles"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMergeOfMergedFiles_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockMergeOfMergedFiles_cfg.py" $?

  echo "testProcessBlockNOMergeOfMergedFiles"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNOMergeOfMergedFiles_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockNOMergeOfMergedFiles_cfg.py" $?

  echo "testProcessBlockRead2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockRead2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockRead2_cfg.py" $?

  echo "testProcessBlockSubProcess"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcess_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockSubProcess_cfg.py" $?

  echo "testProcessBlockSubProcessRead1"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcessRead1_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockSubProcessRead1_cfg.py" $?

  echo "testProcessBlockSubProcessRead2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcessRead2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockSubProcessRead2_cfg.py" $?

  echo "testProcessBlockSubProcessLooper"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcessLooper_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockSubProcessLooper_cfg.py" $?

  echo "testProcessBlock5"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock5_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlock5_cfg.py" $?

  echo "testProcessBlockMerge3"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMerge3_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockMerge3_cfg.py" $?

  echo "testProcessBlockMergeOfMergedFiles2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMergeOfMergedFiles2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockMergeOfMergedFiles2_cfg.py" $?

  echo "testProcessBlockDropOnInput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockDropOnInput_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockDropOnInput_cfg.py" $?

  echo "testProcessBlockThreeFileInput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockThreeFileInput_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockThreeFileInput_cfg.py" $?

  echo "testProcessBlockReadThreeFileInput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockReadThreeFileInput_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockReadThreeFileInput_cfg.py" $?

  echo "testLooperEventNavigation2"
  cmsRun -p ${LOCAL_TEST_DIR}/testLooperEventNavigation2_cfg.py < ${LOCAL_TEST_DIR}/testLooperEventNavigation2.txt > /dev/null 2>&1 || die "cmsRun testLooperEventNavigation2_cfg.py" $?

  echo "testLooperEventNavigation3"
  cmsRun -p ${LOCAL_TEST_DIR}/testLooperEventNavigation3_cfg.py < ${LOCAL_TEST_DIR}/testLooperEventNavigation3.txt > /dev/null 2>&1 || die "cmsRun testLooperEventNavigation3_cfg.py" $?

  echo "testProcessBlockDropOnOutput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockDropOnOutput_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockDropOnOutput_cfg.py" $?

  echo "testProcessBlockDropOnOutput2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockDropOnOutput2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockDropOnOutput2_cfg.py" $?

  echo "testProcessBlockReadDropOnOutput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockReadDropOnOutput_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockReadDropOnOutput_cfg.py" $?

  echo "testProcessBlockReadDropOnOutput2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockReadDropOnOutput2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockReadDropOnOutput2_cfg.py" $?

  # The next three tests would be relevant if we disabled the strict merging requirement
  # in ProductRegistry.cc for ProcessBlock products (a one line code change). As long
  # as we always enforce the strict merging requirement these tests will fail, but they
  # would be useful if we decide to allow that requirement to be disabled in the future.
  # I ran them manually with the ProductRegistry.cc modified  to disable the requirement
  # and in May 2021 these tests passed.

  #echo "testProcessBlockNonStrict"
  #cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNonStrict_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockNonStrict_cfg.py" $?

  #echo "testProcessBlockNonStrict2"
  #cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNonStrict2_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockNonStrict2_cfg.py" $?

  #echo "testProcessBlockNonStrict3"
  #cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNonStrict3_cfg.py > /dev/null 2>&1 || die "cmsRun testProcessBlockNonStrict3_cfg.py" $?

  rm testProcessBlock1ContentsM.txt
  rm testProcessBlock1ContentsE.txt
  rm testProcessBlock2DroppedContentsM.txt
  rm testProcessBlock2DroppedContentsE.txt
  rm testProcessBlockMContentsM.txt
  rm testProcessBlockMContentsE.txt
  rm testProcessBlockRead.log

  rm testProcessBlock1.root
  rm testProcessBlock2.root
  rm testProcessBlock3.root
  rm testProcessBlock4.root
  rm testProcessBlockMerge.root
  rm testProcessBlockTest.root
  rm testProcessBlockRead.root
  rm testProcessBlock2Dropped.root
  rm testProcessBlockFailMerge.root
  rm testProcessBlockMerge2.root
  rm testProcessBlockMergeOfMergedFiles.root
  rm testProcessBlockNOMergeOfMergedFiles.root
  rm testProcessBlockRead2.root
  rm testProcessBlockSubProcessTest.root
  rm testProcessBlockSubProcessRead.root
  rm testProcessBlockSubProcessReadAgain.root
  rm testProcessBlockSubProcessRead1.root
  rm testProcessBlockSubProcessRead2.root
  rm testProcessBlockSubProcessLooperTest.root
  rm testProcessBlockSubProcessLooperRead.root
  rm testProcessBlockSubProcessLooperReadAgain.root
  rm testProcessBlock5.root
  rm testProcessBlockMerge3.root
  rm testProcessBlockMergeOfMergedFiles2.root
  rm testProcessBlockDropOnInput.root
  rm testProcessBlockThreeFileInput.root
  rm testProcessBlockReadThreeFileInput.root
  rm testProcessBlockDropOnOutput.root
  rm testProcessBlockDropOnOutput2.root
  rm testProcessBlockDropOnOutput2_2.root
  rm testProcessBlockReadDropOnOutput.root
  rm testProcessBlockReadDropOnOutput2.root
popd

exit 0
