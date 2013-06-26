#!/bin/bash

# I was not successful figuring out how to run this shell script outside
# of the "scramv1 b runtests" command, but one can run the executable
# run by this script using the values for the two variables (modified for
# each test release) along with the commands in the shell script below.
#LOCAL_TMP_DIR=/uscms_data/d1/wdd/CMSSW_1_8_0_pre2/tmp/slc4_ia32_gcc345
#LOCAL_TEST_DIR=/uscms_data/d1/wdd/CMSSW_1_8_0_pre2/src/FWCore/Framework/test

exe=${LOCALRT}/test/${SCRAM_ARCH}/TestFWCoreFrameworkStatemachine
input=${LOCAL_TEST_DIR}/unit_test_outputs/statemachine_
output=statemachine_output_
reference_output=${LOCAL_TEST_DIR}/unit_test_outputs/statemachine_output_

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  for i in 1 2 5 6 7 8 9 10 11
  do

    ${exe} -i ${input}${i}.txt -o ${output}${i}.txt || die "TestFWCoreFrameworkStatemachine with input ${i}" $?
    diff ${reference_output}${i}.txt ${output}${i}.txt || die "comparing ${output}${i}.txt" $?  

  done

  for i in 3 4
  do

    ${exe} -i ${input}${i}.txt -o ${output}${i}.txt || die "TestFWCoreFrameworkStatemachine with input ${i}" $?
    diff ${reference_output}${i}.txt ${output}${i}.txt || die "comparing ${output}${i}.txt" $?  

  done

  for i in 12
  do

    ${exe} -i ${input}${i}.txt -o ${output}${i}.txt || die "TestFWCoreFrameworkStatemachine with input ${i}" $?
    diff ${reference_output}${i}.txt ${output}${i}.txt || die "comparing ${output}${i}.txt" $?  

  done

popd

exit 0
