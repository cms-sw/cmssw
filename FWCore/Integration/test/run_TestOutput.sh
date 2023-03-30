#!/bin/bash

test=testOutput

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  echo "testOutput1"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}1_cfg.py 2> testOutput1.log || die "cmsRun ${test}1_cfg.py" $?

  # Check that all the transitions that were supposed to occur
  # in a global output module actually did occur
  grep "global write event" testOutput1.log > /dev/null || die "grep failed to find 'global write event'" $?
  grep "global writeLuminosityBlock" testOutput1.log > /dev/null || die "grep failed to find 'global writeLuminosityBlock'" $?
  grep "global writeRun" testOutput1.log > /dev/null || die "grep failed to find 'global writeRun'" $?
  grep "global writeProcessBlock" testOutput1.log > /dev/null || die "grep failed to find 'global writeProcessBlock'" $?
  grep "global respondToOpenInputFile" testOutput1.log > /dev/null || die "grep failed to find 'global respondToOpenInputFile'" $?
  grep "global respondToCloseInputFile" testOutput1.log > /dev/null || die "grep failed to find 'global respondToCloseInputFile'" $?
  grep "global globalBeginRun" testOutput1.log > /dev/null || die "grep failed to find 'global globalBeginRun'" $?
  grep "global globalEndRun" testOutput1.log > /dev/null || die "grep failed to find 'global globalEndRun'" $?
  grep "global globalBeginLuminosityBlock" testOutput1.log > /dev/null || die "grep failed to find 'global globalBeginLuminosityBlock'" $?
  grep "global globalEndLuminosityBlock" testOutput1.log > /dev/null || die "grep failed to find 'global globalEndLuminosityBlock'" $?

  # Check the branch ID for the EDAliases was placed correctly in the BranchIDLists
  grep "global branchID 3673681161" testOutput1.log > /dev/null || die "grep failed to find 'global branchID 3673681161'" $?

  # Repeat checks for the limited module
  grep "limited write event" testOutput1.log > /dev/null || die "grep failed to find 'limited write event'" $?
  grep "limited writeLuminosityBlock" testOutput1.log > /dev/null || die "grep failed to find 'limited writeLuminosityBlock'" $?
  grep "limited writeRun" testOutput1.log > /dev/null || die "grep failed to find 'limited writeRun'" $?
  grep "limited writeProcessBlock" testOutput1.log > /dev/null || die "grep failed to find 'limited writeProcessBlock'" $?
  grep "limited respondToOpenInputFile" testOutput1.log > /dev/null || die "grep failed to find 'limited respondToOpenInputFile'" $?
  grep "limited respondToCloseInputFile" testOutput1.log > /dev/null || die "grep failed to find 'limited respondToCloseInputFile'" $?
  grep "limited globalBeginRun" testOutput1.log > /dev/null || die "grep failed to find 'limited globalBeginRun'" $?
  grep "limited globalEndRun" testOutput1.log > /dev/null || die "grep failed to find 'limited globalEndRun'" $?
  grep "limited globalBeginLuminosityBlock" testOutput1.log > /dev/null || die "grep failed to find 'limited globalBeginLuminosityBlock'" $?
  grep "limited globalEndLuminosityBlock" testOutput1.log > /dev/null || die "grep failed to find 'limited globalEndLuminosityBlock'" $?
  grep "limited branchID 3673681161" testOutput1.log > /dev/null || die "grep failed to find 'limited branchID 3673681161'" $?

  # Above we tested using EmptySource. Repeat reading a file using PoolSource
  echo "testOutput2"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}2_cfg.py 2> testOutput2.log || die "cmsRun ${test}2_cfg.py" $?

  grep "global write event" testOutput2.log > /dev/null || die "grep failed to find 'global write event'" $?
  grep "global writeLuminosityBlock" testOutput2.log > /dev/null || die "grep failed to find 'global writeLuminosityBlock'" $?
  grep "global writeRun" testOutput2.log > /dev/null || die "grep failed to find 'global writeRun'" $?
  grep "global writeProcessBlock" testOutput2.log > /dev/null || die "grep failed to find 'global writeProcessBlock'" $?
  grep "global respondToOpenInputFile" testOutput2.log > /dev/null || die "grep failed to find 'global respondToOpenInputFile'" $?
  grep "global respondToCloseInputFile" testOutput2.log > /dev/null || die "grep failed to find 'global respondToCloseInputFile'" $?
  grep "global globalBeginRun" testOutput2.log > /dev/null || die "grep failed to find 'global globalBeginRun'" $?
  grep "global globalEndRun" testOutput2.log > /dev/null || die "grep failed to find 'global globalEndRun'" $?
  grep "global globalBeginLuminosityBlock" testOutput2.log > /dev/null || die "grep failed to find 'global globalBeginLuminosityBlock'" $?
  grep "global globalEndLuminosityBlock" testOutput2.log > /dev/null || die "grep failed to find 'global globalEndLuminosityBlock'" $?
  grep "global branchID 4057644746" testOutput2.log > /dev/null || die "grep failed to find 'global branchID 4057644746'" $?

  grep "limited write event" testOutput2.log > /dev/null || die "grep failed to find 'limited write event'" $?
  grep "limited writeLuminosityBlock" testOutput2.log > /dev/null || die "grep failed to find 'limited writeLuminosityBlock'" $?
  grep "limited writeRun" testOutput2.log > /dev/null || die "grep failed to find 'limited writeRun'" $?
  grep "limited writeProcessBlock" testOutput2.log > /dev/null || die "grep failed to find 'limited writeProcessBlock'" $?
  grep "limited respondToOpenInputFile" testOutput2.log > /dev/null || die "grep failed to find 'limited respondToOpenInputFile'" $?
  grep "limited respondToCloseInputFile" testOutput2.log > /dev/null || die "grep failed to find 'limited respondToCloseInputFile'" $?
  grep "limited globalBeginRun" testOutput2.log > /dev/null || die "grep failed to find 'limited globalBeginRun'" $?
  grep "limited globalEndRun" testOutput2.log > /dev/null || die "grep failed to find 'limited globalEndRun'" $?
  grep "limited globalBeginLuminosityBlock" testOutput2.log > /dev/null || die "grep failed to find 'limited globalBeginLuminosityBlock'" $?
  grep "limited globalEndLuminosityBlock" testOutput2.log > /dev/null || die "grep failed to find 'limited globalEndLuminosityBlock'" $?
  grep "limited branchID 4057644746" testOutput2.log > /dev/null || die "grep failed to find 'limited branchID 4057644746'" $?

exit 0
