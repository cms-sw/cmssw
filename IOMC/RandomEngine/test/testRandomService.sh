#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

test=testRandomService

pushd ${LOCAL_TMP_DIR}

  # first test that old and new interfaces give the same results
  echo RandomNumberGeneratorService test old format
  rm -f testRandomService.txt
  cmsRun ${LOCAL_TEST_DIR}/oldStyle_cfg.py > testRandomServiceOldDump.txt || die "cmsRun oldStyle_cfg.py" $?
  mv testRandomService.txt testRandomServiceOld.txt

  echo RandomNumberGeneratorService test new format
  cmsRun ${LOCAL_TEST_DIR}/newStyle_cfg.py > testRandomServiceNewDump.txt || die "cmsRun newStyle_cfg.py" $?
  mv testRandomService.txt testRandomServiceNew.txt

  # Here are the comparisons of the results using the old and new interfaces
  diff testRandomServiceOld.txt testRandomServiceNew.txt || die "comparing testRandomServiceOld.txt testRandomServiceNew.txt" $?
  diff testRandomServiceOldDump.txt testRandomServiceNewDump.txt || die "comparing testRandomServiceOldDump.txt testRandomServiceNewDump.txt" $?

  # From this point on use exclusively the new interface in the tests

  echo " "
  echo "RandomNumberGeneratorService 1"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}1_cfg.py > testRandomService1Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}1_cfg.py" $?
  diff testRandomService1Dump.txt ${LOCAL_TEST_DIR}/unit_test_outputs/testRandomService1Dump.txt || die "comparing testRandomService1Dump.txt" $?
  mv ${test}.txt ${test}1.txt
  # Note that in the reference I verified that the sequence of numbers in the
  # events continues over the second beginLuminosityBlock as if it was not there.
  # The random numbers in the second beginLuminosityBlock are the same as the first one.
  diff ${test}1.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${test}1.txt || die "comparing ${test}1.txt" $?

  echo " "
  echo "RandomNumberGeneratorService 2"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}2_cfg.py > testRandomService2Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}2_cfg.py" $?
  mv ${test}.txt ${test}2.txt
  diff ${test}2.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${test}2.txt || die "comparing ${test}2.txt" $?

  echo " "
  echo "RandomNumberGeneratorService 3"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}3_cfg.py > testRandomService3Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}3_cfg.py" $?
  mv ${test}.txt ${test}3.txt
  diff ${test}3.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${test}3.txt || die "comparing ${test}3.txt" $?

  echo " "
  echo "RandomNumberGeneratorService merge"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge1_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Merge1_cfg.py" $?

  echo " "
  echo "RandomNumberGeneratorService test 1"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Test1_cfg.py > testRandomServiceTest1Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test1_cfg.py" $?
  mv ${test}.txt ${test}Test1.txt
  diff ${test}Test1.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${test}Test1.txt || die "comparing ${test}Test1.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test 2"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Test2_cfg.py > testRandomServiceTest2Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test2_cfg.py" $?
  mv ${test}.txt ${test}Test2.txt
  diff ${test}Test2.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${test}Test2.txt || die "comparing ${test}Test2.txt" $?

  echo " "
  echo "RandomNumberGeneratorService multiprocess"
  echo "=============================================="
  rm -f child0FirstEvent.txt
  rm -f child1FirstEvent.txt
  rm -f child2FirstEvent.txt
  rm -f child0LastEvent.txt
  rm -f child1LastEvent.txt
  rm -f child2LastEvent.txt
  rm -f origChild0LastEvent.txt
  rm -f origChild1LastEvent.txt
  rm -f origChild2LastEvent.txt
  rm -f testRandomServiceL1E1.txt
  rm -f testRandomServiceL1E2.txt
  rm -f testRandomServiceL1E3.txt
  rm -f testRandomServiceL2E4.txt
  rm -f testRandomServiceL2E5.txt
  rm -f StashStateFork.data_0
  rm -f StashStateFork.data_1
  rm -f StashStateFork.data_2

  cmsRun ${LOCAL_TEST_DIR}/testMultiProcess_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/testMultiProcess_cfg.py" $?

  # We check if the file exists because it is possible for the multiprocessing not
  # to give a child process any events
  if [ -f child0FirstEvent.txt ]
  then
    echo compare child0FirstEvent.txt
    diff child0FirstEvent.txt ${LOCAL_TEST_DIR}/unit_test_outputs/child0FirstEvent.txt || die "comparing child0FirstEvent.txt" $?
  fi
  if [ -f child1FirstEvent.txt ]
  then
    echo compare child1FirstEvent.txt
    diff child1FirstEvent.txt ${LOCAL_TEST_DIR}/unit_test_outputs/child1FirstEvent.txt || die "comparing child1FirstEvent.txt" $?
  fi
  if [ -f child2FirstEvent.txt ]
  then
    echo compare child2FirstEvent.txt
    diff child2FirstEvent.txt ${LOCAL_TEST_DIR}/unit_test_outputs/child2FirstEvent.txt || die "comparing child2FirstEvent.txt" $?
  fi

  mv child0LastEvent.txt origChild0LastEvent.txt
  mv child1LastEvent.txt origChild1LastEvent.txt
  mv child2LastEvent.txt origChild2LastEvent.txt
  rm -f testRandomServiceL1E1.txt
  rm -f testRandomServiceL1E2.txt
  rm -f origTestRandomServiceL1E3.txt
  rm -f origTestRandomServiceL2E4.txt
  rm -f origTestRandomServiceL2E5.txt
  mv testRandomServiceL1E3.txt origTestRandomServiceL1E3.txt
  mv testRandomServiceL2E4.txt origTestRandomServiceL2E4.txt
  mv testRandomServiceL2E5.txt origTestRandomServiceL2E5.txt

  echo " "
  echo "RandomNumberGeneratorService merge 2"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge2_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Merge2_cfg.py" $?

  echo " "
  echo "RandomNumberGeneratorService test 3"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Test3_cfg.py > testRandomServiceTest3Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test3_cfg.py" $?
  diff testRandomServiceL1E3.txt origTestRandomServiceL1E3.txt || die "comparing testRandomServiceL1E3.txt" $?
  diff testRandomServiceL2E4.txt origTestRandomServiceL2E4.txt || die "comparing testRandomServiceL2E4.txt" $?
  diff testRandomServiceL2E5.txt origTestRandomServiceL2E5.txt || die "comparing testRandomServiceL2E5.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test 4"
  echo "=============================================="
  rm -f child0LastEvent.txt

  # We use this if block because it is possible for the multiprocessing not
  # to give a child any events so we look for the last child with events
  # and replay the random numbers for that child.
  if [ -f StashStateFork.data_2 ]
  then
    echo running cmsRun with StashStateFork.data_2
    cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py" $?
    diff child0LastEvent.txt origChild2LastEvent.txt  || die "comparing test 4 .txt file" $?
  elif [ -f StashStateFork.data_1 ]
  then
    echo running cmsRun with StashStateFork.data_1
    mv StashStateFork.data_1 StashStateFork.data_2
    cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py" $?
    diff child0LastEvent.txt origChild1LastEvent.txt  || die "comparing test 4 .txt file" $?
  elif [ -f StashStateFork.data_0 ]
  then
    echo running cmsRun with StashStateFork.data_0
    mv StashStateFork.data_0 StashStateFork.data_2
    cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py" $?
    diff child0LastEvent.txt origChild0LastEvent.txt  || die "comparing test 4 .txt file" $?
  else
    echo Error: Text file containing states not found
    exit 1
  fi

popd

exit 0
