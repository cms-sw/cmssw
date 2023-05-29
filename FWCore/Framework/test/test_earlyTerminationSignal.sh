#!/bin/bash

# Pass in name and status

function die { echo $1: status $2 ;  exit $2; }

function test_failure { 
    if [ "$2" != "90" ]
     then
       echo $1: status $2; exit $2;
    fi
}

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
echo "running cmsRun testEarlyTerminationSignal_cfg.py"
(cmsRun ${LOCAL_TEST_DIR}/testEarlyTerminationSignal_cfg.py 2>&1 | grep -q 'early termination of event: stream = 0 run = 1 lumi = 1 event = 10 : time = 50000001') || die "Early termination signal failed" $?

echo "running cmsRun test_dependentPathsAndExceptions_cfg.py"
(cmsRun ${LOCAL_TEST_DIR}/test_dependentPathsAndExceptions_cfg.py 2>&1 | grep -q "Intentional 'NotFound' exception for testing purposes") || die "dependent Paths and Exceptions failed" $?

echo "running cmsRun test_dependentRunDataAndException_cfg.py"
(cmsRun ${LOCAL_TEST_DIR}/test_dependentRunDataAndException_cfg.py 2>&1 | grep -q "Intentional 'NotFound' exception for testing purposes") || die "dependent Run data and Exceptions failed" $?

echo "running cmsRun test_exceptionAtGlobalBeginRun_cfg.py"
(cmsRun ${LOCAL_TEST_DIR}/test_exceptionAtGlobalBeginRun_cfg.py 2>&1 | grep -q -v "An exception of category 'transitions' occurred") || die "exception at globalBeginRun failed" $?

echo "running cmsRun  test_exceptionInShortLumi_cfg.py"
cmsRun ${LOCAL_TEST_DIR}/test_exceptionInShortLumi_cfg.py; test_failure "test_exceptionInShortLumi_cfg.py failed" $?
