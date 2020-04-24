#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

test=testRandomService

pushd ${LOCAL_TMP_DIR}


  echo " "
  echo "RandomNumberGeneratorService 1"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${test}1_cfg.py > testRandomService1Dump.txt || die "cmsRun ${LOCAL_TEST_DIR}/${test}1_cfg.py" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testRandomService1Dump.txt  testRandomService1Dump.txt || die "comparing testRandomService1Dump.txt" $?
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
  diff -I "TrackTSelector" ${LOCAL_TEST_DIR}/unit_test_outputs/testMultiStreamDump.txt  testMultiStreamDump.txt || die "comparing testMultiStreamDump.txt" $?

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

popd

exit 0
