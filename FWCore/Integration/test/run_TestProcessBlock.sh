#!/bin/bash
set -x
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

function die { echo Failure $1: status $2 ; exit $2 ; }

# The tests executed by this bash script are all related and
# it seemed clearest to include them all in the same file.
# These tests are divided into distinct groups. Each time
# the script runs, it will execute only one group of tests.
# The script requires that its first command line argument
# specifies the group to be run. The "if" conditional statements
# below implement this. The BuildFile directs scram to run
# this script once for each group when unit tests are run.
# The BuildFile also specifies the dependencies between the
# groups. In some cases, one group cannot run until another
# group of tests has finished. The purpose of this is to
# allow maximum concurrency while running the tests so the
# tests can run faster.

if [ $1 -eq 1 ]
then
  echo "testProcessBlock1"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock1_cfg.py &> testProcessBlock1.log || die "cmsRun testProcessBlock1_cfg.py" $?

  # The MetaData ProcessBlock branch and the TTree should exist to hold the ProcessBlock
  # data. The Events branch should not exist because there were not any ProcessBlock branches
  # saved from an input file. Test that here:
  edmFileUtil -l -t MetaData -P file:testProcessBlock1.root > testProcessBlock1ContentsM.txt
  grep "Branch.* ProcessBlockHelper " testProcessBlock1ContentsM.txt || die "Check for existence of ProcessBlockHelper branch" $?
  grep "TTree.*ProcessBlocksPROD1" testProcessBlock1ContentsM.txt || die "Check for existence of ProcessBlocksPROD1 TTree" $?
  edmFileUtil -t Events -P file:testProcessBlock1.root > testProcessBlock1ContentsE.txt
  grep "Branch.* EventToProcessBlockIndexes " testProcessBlock1ContentsE.txt && die "Check for non-existence of eventToProcessBlockIndexes branch" 1
fi

if [ $1 -eq 2 ]
then
  echo "testProcessBlock2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock2_cfg.py &> testProcessBlock2.log || die "cmsRun testProcessBlock2_cfg.py" $?
fi

if [ $1 -eq 3 ]
then
  echo "testProcessBlock3"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock3_cfg.py &> testProcessBlock3.log || die "cmsRun testProcessBlock3_cfg.py" $?
fi

if [ $1 -eq 4 ]
then
  echo "testProcessBlock4"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock4_cfg.py &> testProcessBlock4.log || die "cmsRun testProcessBlock4_cfg.py" $?
fi

if [ $1 -eq 5 ]
then
  echo "testProcessBlockMerge"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMerge_cfg.py &> testProcessBlockMerge.log || die "cmsRun testProcessBlockMerge_cfg.py" $?

  # The ProcessBlock Branches and TTrees should exist in this case. Test that here:
  edmFileUtil -l -t MetaData -P file:testProcessBlockMerge.root > testProcessBlockMContentsM.txt
  grep "Branch.* ProcessBlockHelper " testProcessBlockMContentsM.txt || die "Check for existence of ProcessBlockHelper branch" $?
  grep "TTree.*ProcessBlocksPROD1" testProcessBlockMContentsM.txt || die "Check for existence of ProcessBlocksPROD1 TTree" $?
  grep "TTree.*ProcessBlocksMERGE" testProcessBlockMContentsM.txt || die "Check for existence of ProcessBlocksMERGE TTree" $?
  edmFileUtil -t Events -P file:testProcessBlockMerge.root > testProcessBlockMContentsE.txt
  grep "Branch.* EventToProcessBlockIndexes " testProcessBlockMContentsE.txt || die "Check for existence of eventToProcessBlockIndexes branch" $?
fi

if [ $1 -eq 6 ]
then
  echo "testProcessBlockTEST"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockTEST_cfg.py &> testProcessBlockTEST.log || die "cmsRun testProcessBlockTEST_cfg.py" $?

  echo "testProcessBlockRead"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockRead_cfg.py &> testProcessBlockRead.log || die "cmsRun testProcessBlockRead_cfg.py" $?
  grep "InputProcessBlockIntAnalyzer::accessInputProcessBlock" testProcessBlockRead.log || die "Check that InputProcessBlockIntAnalyzer::accessInputProcessBlock was called" $?
  grep "InputProcessBlockIntFilter::accessInputProcessBlock" testProcessBlockRead.log || die "Check that InputProcessBlockIntFilter::accessInputProcessBlock was called" $?
  grep "InputProcessBlockIntProducer::accessInputProcessBlock" testProcessBlockRead.log || die "Check that InputProcessBlockIntProducer::accessInputProcessBlock was called" $?
fi

if [ $1 -eq 7 ]
then
  echo "testProcessBlock2Dropped"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock2Dropped_cfg.py &> testProcessBlock2Dropped.log || die "cmsRun testProcessBlock2Dropped_cfg.py" $?

  # The ProcessBlock Branches and TTrees should not exist in this case because
  # all the ProcessBlock products are dropped. Test that here:
  edmFileUtil -l -t MetaData -P file:testProcessBlock2Dropped.root > testProcessBlock2DroppedContentsM.txt
  grep "Branch.* ProcessBlockHelper " testProcessBlock2DroppedContentsM.txt && die "Check for non-existence of ProcessBlockHelper branch" 1
  grep "TTree.*ProcessBlocksPROD1" testProcessBlock2DroppedContentsM.txt && die "Check for non-existence of ProcessBlocksPROD1 TTree" 1
  edmFileUtil -t Events -P file:testProcessBlock2Dropped.root > testProcessBlock2DroppedContentsE.txt
  grep "Branch.* EventToProcessBlockIndexes " testProcessBlock2DroppedContentsE.txt && die "Check for non-existence of eventToProcessBlockIndexes branch" 1
fi

if [ $1 -eq 8 ]
then
  # This one intentionally fails because the product content of the
  # files does not match (strict merging requirements for ProcessBlocks)
  echo "testProcessBlockFailMerge"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockFailMerge_cfg.py &> testProcessBlockFailMerge.log && die "cmsRun testProcessBlockFailMerge_cfg.py" 1
fi

if [ $1 -eq 9 ]
then
  echo "testProcessBlockMerge2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMerge2_cfg.py &> testProcessBlockMerge2.log || die "cmsRun testProcessBlockMerge2_cfg.py" $?
fi

if [ $1 -eq 10 ]
then
  echo "testProcessBlockMergeOfMergedFiles"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMergeOfMergedFiles_cfg.py &> testProcessBlockMergeOfMergedFiles.log || die "cmsRun testProcessBlockMergeOfMergedFiles_cfg.py" $?
fi

if [ $1 -eq 11 ]
then
  echo "testProcessBlockNOMergeOfMergedFiles"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNOMergeOfMergedFiles_cfg.py &> testProcessBlockNOMergeOfMergedFiles.log || die "cmsRun testProcessBlockNOMergeOfMergedFiles_cfg.py" $?
fi

if [ $1 -eq 12 ]
then
  echo "testProcessBlockRead2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockRead2_cfg.py &> testProcessBlockRead2.log || die "cmsRun testProcessBlockRead2_cfg.py" $?
fi

if [ $1 -eq 14 ]
then
  echo "testProcessBlockSubProcess"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcess_cfg.py &> testProcessBlockSubProcess.log || die "cmsRun testProcessBlockSubProcess_cfg.py" $?
fi

if [ $1 -eq 15 ]
then
  echo "testProcessBlockSubProcessRead1"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcessRead1_cfg.py &> testProcessBlockSubProcessRead1.log || die "cmsRun testProcessBlockSubProcessRead1_cfg.py" $?
fi

if [ $1 -eq 16 ]
then
  echo "testProcessBlockSubProcessRead2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcessRead2_cfg.py &> testProcessBlockSubProcessRead2.log || die "cmsRun testProcessBlockSubProcessRead2_cfg.py" $?
fi

if [ $1 -eq 17 ]
then
  echo "testProcessBlockSubProcessLooper"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockSubProcessLooper_cfg.py &> testProcessBlockSubProcessLooper.log || die "cmsRun testProcessBlockSubProcessLooper_cfg.py" $?
fi

if [ $1 -eq 18 ]
then
  echo "testProcessBlock5"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlock5_cfg.py &> testProcessBlock5.log || die "cmsRun testProcessBlock5_cfg.py" $?

  echo "testProcessBlockMerge3"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMerge3_cfg.py &> testProcessBlockMerge3.log || die "cmsRun testProcessBlockMerge3_cfg.py" $?

  echo "testProcessBlockMergeOfMergedFiles2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockMergeOfMergedFiles2_cfg.py &> testProcessBlockMergeOfMergedFiles2.log || die "cmsRun testProcessBlockMergeOfMergedFiles2_cfg.py" $?
fi

if [ $1 -eq 19 ]
then
  echo "testProcessBlockDropOnInput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockDropOnInput_cfg.py &> testProcessBlockDropOnInput.log || die "cmsRun testProcessBlockDropOnInput_cfg.py" $?
fi

if [ $1 -eq 20 ]
then
  echo "testProcessBlockThreeFileInput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockThreeFileInput_cfg.py &> testProcessBlockThreeFileInput.log || die "cmsRun testProcessBlockThreeFileInput_cfg.py" $?

  echo "testProcessBlockReadThreeFileInput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockReadThreeFileInput_cfg.py &> testProcessBlockReadThreeFileInput.log || die "cmsRun testProcessBlockReadThreeFileInput_cfg.py" $?
fi

if [ $1 -eq 21 ]
then
  echo "testLooperEventNavigation2"
  cmsRun -p ${LOCAL_TEST_DIR}/testLooperEventNavigation2_cfg.py < ${LOCAL_TEST_DIR}/testLooperEventNavigation2.txt &> testLooperEventNavigation2.log || die "cmsRun testLooperEventNavigation2_cfg.py" $?
fi

if [ $1 -eq 22 ]
then
  echo "testLooperEventNavigation3"
  cmsRun -p ${LOCAL_TEST_DIR}/testLooperEventNavigation3_cfg.py < ${LOCAL_TEST_DIR}/testLooperEventNavigation3.txt &> testLooperEventNavigation3.log || die "cmsRun testLooperEventNavigation3_cfg.py" $?
fi

if [ $1 -eq 23 ]
then
  echo "testProcessBlockDropOnOutput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockDropOnOutput_cfg.py &> testProcessBlockDropOnOutput.log || die "cmsRun testProcessBlockDropOnOutput_cfg.py" $?

  echo "testProcessBlockReadDropOnOutput"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockReadDropOnOutput_cfg.py &> testProcessBlockReadDropOnOutput.log || die "cmsRun testProcessBlockReadDropOnOutput_cfg.py" $?
fi

if [ $1 -eq 24 ]
then
  echo "testProcessBlockDropOnOutput2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockDropOnOutput2_cfg.py &> testProcessBlockDropOnOutput2.log || die "cmsRun testProcessBlockDropOnOutput2_cfg.py" $?

  echo "testProcessBlockReadDropOnOutput2"
  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockReadDropOnOutput2_cfg.py &> testProcessBlockReadDropOnOutput2.log || die "cmsRun testProcessBlockReadDropOnOutput2_cfg.py" $?
fi

# The next three tests would be relevant if we disabled the strict merging requirement
# in ProductRegistry.cc for ProcessBlock products (a one line code change). As long
# as we always enforce the strict merging requirement these tests will fail, but they
# would be useful if we decide to allow that requirement to be disabled in the future.
# I ran them manually with the ProductRegistry.cc modified  to disable the requirement
# and in May 2021 these tests passed. In addition to uncommenting the tests here, they
# would also need to be added in the BuildFile with the proper dependency (both 25
# and 26 depend on 19 at the moment)

#if [ $1 -eq 25 ]
#then
#  echo "testProcessBlockNonStrict"
#  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNonStrict_cfg.py &> testProcessBlockNonStrict.log || die "cmsRun testProcessBlockNonStrict_cfg.py" $?
#
#  echo "testProcessBlockNonStrict2"
#  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNonStrict2_cfg.py &> testProcessBlockNonStrict2.log || die "cmsRun testProcessBlockNonStrict2_cfg.py" $?
#fi

#if [ $1 -eq 26 ]
#then
#  echo "testProcessBlockNonStrict3"
#  cmsRun -p ${LOCAL_TEST_DIR}/testProcessBlockNonStrict3_cfg.py &> testProcessBlockNonStrict3.log || die "cmsRun testProcessBlockNonStrict3_cfg.py" $?
#fi

if [ $1 -eq 100 ]
then
  rm testProcessBlock1ContentsM.txt
  rm testProcessBlock1ContentsE.txt
  rm testProcessBlockMContentsM.txt
  rm testProcessBlockMContentsE.txt
  rm testProcessBlock2DroppedContentsM.txt
  rm testProcessBlock2DroppedContentsE.txt

  rm testProcessBlock1.log
  rm testProcessBlock2.log
  rm testProcessBlock3.log
  rm testProcessBlock4.log
  rm testProcessBlockMerge.log
  rm testProcessBlockTEST.log
  rm testProcessBlockRead.log
  rm testProcessBlock2Dropped.log
  rm testProcessBlockFailMerge.log
  rm testProcessBlockMerge2.log
  rm testProcessBlockMergeOfMergedFiles.log
  rm testProcessBlockNOMergeOfMergedFiles.log
  rm testProcessBlockRead2.log
  rm testProcessBlockSubProcess.log
  rm testProcessBlockSubProcessRead1.log
  rm testProcessBlockSubProcessRead2.log
  rm testProcessBlockSubProcessLooper.log
  rm testProcessBlock5.log
  rm testProcessBlockMerge3.log
  rm testProcessBlockMergeOfMergedFiles2.log
  rm testProcessBlockDropOnInput.log
  rm testProcessBlockThreeFileInput.log
  rm testProcessBlockReadThreeFileInput.log
  rm testLooperEventNavigation2.log
  rm testLooperEventNavigation3.log
  rm testProcessBlockDropOnOutput.log
  rm testProcessBlockReadDropOnOutput.log
  rm testProcessBlockDropOnOutput2.log
  rm testProcessBlockReadDropOnOutput2.log

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
  rm testProcessBlockNOMergeOfMergedFiles001.root
  rm testProcessBlockSubProcessLooperRead001.root
  rm testProcessBlockSubProcessLooperRead002.root
  rm testProcessBlockSubProcessLooperReadAgain001.root
  rm testProcessBlockSubProcessLooperReadAgain002.root
  rm testProcessBlockSubProcessLooperTest001.root
  rm testProcessBlockSubProcessLooperTest002.root

  #rm testProcessBlockNonStrict.log
  #rm testProcessBlockNonStrict2.log
  #rm testProcessBlockNonStrict3.log
  #rm testProcessBlockNonStrict.root
  #rm testProcessBlockNonStrict2.root
  #rm testProcessBlockNonStrict3.root

fi

exit 0
