#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

input=${SCRAM_TEST_PATH}/UnpGenEvent10.hepmc

# the same file is given twice, so that the chaining of several input files is
# exercised as well: the two events of the output come from two different files
cmsRun ${SCRAM_TEST_PATH}/testReader3_cfg.py \
  inputFiles=file:${input},file:${input} outputFile=testReader3.root \
  || die "cmsRun testReader3_cfg.py" $?

edmFileUtil -f file:testReader3.root | grep -q " 2 events" \
  || die "testReader3.root does not hold the 2 expected events" $?
