#!/bin/bash

# Capable of running 5 tests, this bash script expects a command line
# argument from 0 to 4 specifying which test to run

LOCAL_TEST_DIR=${CMSSW_BASE}/src/FWCore/Framework/test

function die { echo Failure $1: status $2 ; exit $2 ; }

echo "Running run_testOptions.sh $1"

# Configuration files and expected outputs for the 5 tests
configFiles=("testOptions0_cfg.py" "testOptions1_cfg.py" "testOptions2_cfg.py" "testOptions3_cfg.py" "testOptions4_cfg.py")
expectedStreams=(1 4 4 4 1)
expectedConcurrentLumis=(1 3 2 4 1)
expectedConcurrentIOVs=(1 2 2 4 1)

cmsRun -p ${LOCAL_TEST_DIR}/${configFiles[$1]} >& ${configFiles[$1]}.log || die "cmsRun ${configFiles[$1]}" $?
grep "Number of Streams = ${expectedStreams[$1]}" ${configFiles[$1]}.log || die "Failed number of streams test" $?
grep "Number of Concurrent Lumis = ${expectedConcurrentLumis[$1]}" ${configFiles[$1]}.log  || die "Failed number of concurrent lumis test" $?
grep "Number of Concurrent IOVs = ${expectedConcurrentIOVs[$1]}" ${configFiles[$1]}.log || die "Failed number of concurrent IOVs test" $?

rm ${configFiles[$1]}.log

exit 0
