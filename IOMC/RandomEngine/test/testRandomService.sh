#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

test=testRandomService

pushd ${LOCAL_TMP_DIR}


  echo " "
  echo "RandomNumberGeneratorService 1"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}1_cfg.py > testRandomService1Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}1_cfg.py" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testRandomService1Dump.txt  testRandomService1Dump.txt || die "comparing testRandomService1Dump.txt" $?
  mv ${test}.txt ${test}1.txt
  mv testRandomService_0_t1.txt testRandomService1_0_t1.txt
  mv testRandomService_0_t2.txt testRandomService1_0_t2.txt
  mv testRandomService_0_t3.txt testRandomService1_0_t3.txt
  mv testRandomService_0_t4.txt testRandomService1_0_t4.txt

  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testRandomService1_0_t1.txt testRandomService1_0_t1.txt || die "comparing testRandomService1_0_t1.txt" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testRandomService1_0_t2.txt testRandomService1_0_t2.txt || die "comparing testRandomService1_0_t2.txt" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testRandomService1_0_t3.txt testRandomService1_0_t3.txt || die "comparing testRandomService1_0_t3.txt" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testRandomService1_0_t4.txt testRandomService1_0_t4.txt || die "comparing testRandomService1_0_t4.txt" $?

  echo " "
  echo "RandomNumberGeneratorService 2"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}2_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}2_cfg.py" $?

  echo " "
  echo "RandomNumberGeneratorService 3"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}3_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}3_cfg.py" $?

  echo " "
  echo "RandomNumberGeneratorService merge"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge1_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Merge1_cfg.py" $?

  echo " "
  echo "RandomNumberGeneratorService test 1, replay from event"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Test1_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test1_cfg.py" $?

  echo " "
  echo "RandomNumberGeneratorService test 2, replay from text file"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}Test2_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test2_cfg.py" $?

  echo " "
  echo "RandomNumberGeneratorService multistream"
  echo "=============================================="

  rm -rf testRandomServiceL1E1.txt
  rm -rf testRandomServiceL1E2.txt
  rm -rf testRandomServiceL1E3.txt
  rm -rf testRandomServiceL2E4.txt
  rm -rf testRandomServiceL2E5.txt

  rm -rf stream0LastEvent.txt
  rm -rf stream1LastEvent.txt
  rm -rf stream2LastEvent.txt

  cmsRun ${LOCAL_TEST_DIR}/testMultiStream_cfg.py > testMultiStreamDump.txt || die "cmsRun testMultiStream_cfg.py" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testMultiStreamDump.txt  testMultiStreamDump.txt || die "comparing testMultiStreamDump.txt" $?

  echo " "
  echo "RandomNumberGeneratorService multistream test replay from event"
  echo "=============================================="

  rm -rf replaytestRandomServiceL1E1.txt
  rm -rf replaytestRandomServiceL1E2.txt
  rm -rf replaytestRandomServiceL1E3.txt
  rm -rf replaytestRandomServiceL2E4.txt
  rm -rf replaytestRandomServiceL2E5.txt

  rm -rf replaystream0LastEvent.txt
  rm -rf replaystream1LastEvent.txt
  rm -rf replaystream2LastEvent.txt

  cmsRun ${LOCAL_TEST_DIR}/testMultiStreamReplay1_cfg.py || die "cmsRun testMultiStreamReplay1_cfg.py" $?

  # sort so this does not depend on module execution order
  sort testRandomServiceL1E3.txt > testRandomServiceL1E3.sorted
  sort testRandomServiceL2E4.txt > testRandomServiceL2E4.sorted
  sort testRandomServiceL2E5.txt > testRandomServiceL2E5.sorted

  sort replaytestRandomServiceL1E3.txt > replaytestRandomServiceL1E3.sorted
  sort replaytestRandomServiceL2E4.txt > replaytestRandomServiceL2E4.sorted
  sort replaytestRandomServiceL2E5.txt > replaytestRandomServiceL2E5.sorted

  diff testRandomServiceL1E3.sorted replaytestRandomServiceL1E3.sorted || die "comparing testRandomServiceL1E3.sorted and replaytestRandomServiceL1E3.sorted" $?
  diff testRandomServiceL2E4.sorted replaytestRandomServiceL2E4.sorted || die "comparing testRandomServiceL2E4.sorted and replaytestRandomServiceL2E4.sorted" $?
  diff testRandomServiceL2E5.sorted replaytestRandomServiceL2E5.sorted || die "comparing testRandomServiceL2E5.sorted and replaytestRandomServiceL2E5.sorted" $?

  echo " "
  echo "RandomNumberGeneratorService multistream test replay from text file"
  echo "=============================================="


  # We use this if block because it is possible for the multithreading not
  # to give a stream any events so we look for the last stream with events
  # and replay the random numbers for that stream.
  if [ -f StashStateStream.data_2 ]
  then
    echo running cmsRun with StashStateStream.data_2
    mv stream2LastEvent.txt lastEvent.txt
  elif [ -f StashStateFork.data_1 ]
  then
    echo running cmsRun with StashStateFork.data_1
    mv StashStateFork.data_1 StashStateFork.data_2
    mv stream1LastEvent.txt lastEvent.txt
  elif [ -f StashStateFork.data_0 ]
  then
    echo running cmsRun with StashStateFork.data_0
    mv StashStateFork.data_0 StashStateFork.data_2
    mv stream0LastEvent.txt lastEvent.txt
  else
    echo Error: Text file containing states not found
    exit 1
  fi

  rm -rf replaystream0LastEvent.txt
  rm -rf replaystream1LastEvent.txt
  rm -rf replaystream2LastEvent.txt

  cmsRun ${LOCAL_TEST_DIR}/testMultiStreamReplay2_cfg.py || die "cmsRun testMultiStreamReplay2_cfg.py" $?

  # sort so this does not depend on module execution order
  sort lastEvent.txt > lastEvent.sorted
  sort replaystream0LastEvent.txt > replayLastEvent.sorted

  diff lastEvent.sorted replayLastEvent.sorted || die "comparing files containing random numbers of last event in a stream" $?

  echo " "
  echo "RandomNumberGeneratorService multiprocess"
  echo "=============================================="
  rm -f child0FirstEvent.txt
  rm -f child1FirstEvent.txt
  rm -f child2FirstEvent.txt
  rm -f child0LastEvent.txt
  rm -f child1LastEvent.txt
  rm -f child2LastEvent.txt
  rm -f testRandomServiceL1E1.txt
  rm -f testRandomServiceL1E2.txt
  rm -f testRandomServiceL1E3.txt
  rm -f testRandomServiceL2E4.txt
  rm -f testRandomServiceL2E5.txt
  rm -f origChild0LastEvent.txt
  rm -f origChild1LastEvent.txt
  rm -f origChild2LastEvent.txt
  rm -f StashStateFork.data_0
  rm -f StashStateFork.data_1
  rm -f StashStateFork.data_2

  cmsRun ${LOCAL_TEST_DIR}/testMultiProcess_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/testMultiProcess_cfg.py" $?

  # We check if the file exists because it is possible for the multiprocessing not
  # to give a child process any events
  if [ -f child0FirstEvent.txt ]
  then
    echo compare child0FirstEvent.txt
    sort child0FirstEvent.txt > child0FirstEvent.sorted
    diff ${LOCAL_TEST_DIR}/unit_test_outputs/child0FirstEvent.txt child0FirstEvent.sorted || die "comparing child0FirstEvent.txt" $?
    mv child0LastEvent.txt origChild0LastEvent.txt
  fi
  if [ -f child1FirstEvent.txt ]
  then
    echo compare child1FirstEvent.txt
    sort child1FirstEvent.txt > child1FirstEvent.sorted
    diff ${LOCAL_TEST_DIR}/unit_test_outputs/child1FirstEvent.txt child1FirstEvent.sorted || die "comparing child1FirstEvent.txt" $?
    mv child1LastEvent.txt origChild1LastEvent.txt
  fi
  if [ -f child2FirstEvent.txt ]
  then
    echo compare child2FirstEvent.txt
    sort child2FirstEvent.txt > child2FirstEvent.sorted
    diff ${LOCAL_TEST_DIR}/unit_test_outputs/child2FirstEvent.txt child2FirstEvent.sorted || die "comparing child2FirstEvent.txt" $?
    mv child2LastEvent.txt origChild2LastEvent.txt
  fi

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
  cmsRun ${LOCAL_TEST_DIR}/${test}Test3_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test3_cfg.py" $?

  sort testRandomServiceL1E3.txt > testRandomServiceL1E3.sorted
  sort testRandomServiceL2E4.txt > testRandomServiceL2E4.sorted
  sort testRandomServiceL2E5.txt > testRandomServiceL2E5.sorted

  sort origTestRandomServiceL1E3.txt > origTestRandomServiceL1E3.sorted
  sort origTestRandomServiceL2E4.txt > origTestRandomServiceL2E4.sorted
  sort origTestRandomServiceL2E5.txt > origTestRandomServiceL2E5.sorted

  diff testRandomServiceL1E3.sorted origTestRandomServiceL1E3.sorted || die "comparing testRandomServiceL1E3.sorted" $?
  diff testRandomServiceL2E4.sorted origTestRandomServiceL2E4.sorted || die "comparing testRandomServiceL2E4.sorted" $?
  diff testRandomServiceL2E5.sorted origTestRandomServiceL2E5.sorted || die "comparing testRandomServiceL2E5.sorted" $?

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

    sort child0LastEvent.txt > child0LastEvent.sorted
    sort origChild2LastEvent.txt > origChild2LastEvent.sorted
    diff child0LastEvent.sorted origChild2LastEvent.sorted  || die "Test 4 text file comparison failed" $?
  elif [ -f StashStateFork.data_1 ]
  then
    echo running cmsRun with StashStateFork.data_1
    mv StashStateFork.data_1 StashStateFork.data_2
    cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py" $?

    sort child0LastEvent.txt > child0LastEvent.sorted
    sort origChild1LastEvent.txt > origChild1LastEvent.sorted
    diff child0LastEvent.sorted origChild1LastEvent.sorted  || die "Test 4 text file comparison failed" $?
  elif [ -f StashStateFork.data_0 ]
  then
    echo running cmsRun with StashStateFork.data_0
    mv StashStateFork.data_0 StashStateFork.data_2
    cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py || die "cmsRun ${LOCAL_TEST_DIR}/${test}Test4_cfg.py" $?

    sort child0LastEvent.txt > child0LastEvent.sorted
    sort origChild0LastEvent.txt > origChild0LastEvent.sorted
    diff child0LastEvent.sorted origChild0LastEvent.sorted  || die "Test 4 text file comparison failed" $?
  else
    echo Error: Text file containing states not found
    exit 1
  fi

popd

exit 0
