#!/bin/bash -x

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
source "${LOCAL_TEST_DIR}/help_cmsRun_tests.sh"

# test different config_file suffix (not .py)
cp ${LOCAL_TEST_DIR}/test_argparse.py test_argparse.notpy
doTest 1 "cmsRun -n 1 test_argparse.notpy" "" "TestArgParse"

# test dash in config_file name
cp -- ${LOCAL_TEST_DIR}/test_argparse.py -test_argparse.py
doTest 2 "cmsRun -n 1 -- -test_argparse.py" "" "TestArgParse"

# do these manually because quote nesting becomes a nightmare / perhaps actually impossible

# test config as command line input
TEST=3
LOG="log_test$TEST.log"
CONFIG_INPUT="import FWCore.ParameterSet.Config as cms; process = cms.Process('Test'); process.source=cms.Source('EmptySource'); process.maxEvents.input=10; print('Test3')"
cmsRun -c "$CONFIG_INPUT" >& $LOG || die "Test $TEST: failure running cmsRun -c \"${CONFIG_INPUT}\""
(grep -qF "Test3" $LOG) || die "Test $TEST: incorrect output from cmsRun -c \"${CONFIG_INPUT}\""

# test command line input + config_file
TEST=4
LOG="log_test$TEST.log"
CONFIG_INPUT="import FWCore.ParameterSet.Config as cms; process = cms.Process('Test'); process.source=cms.Source('EmptySource'); process.maxEvents.input=10"
cmsRun -c "$CONFIG_INPUT" ${LOCAL_TEST_DIR}/test_argparse.py >& $LOG && die "Test $TEST: no error from cmsRun -c \"${CONFIG_INPUT}\" ${LOCAL_TEST_DIR}/test_argparse.py"
(grep -qF "cannot use '-c [command line input]' with 'config_file'" $LOG) || die "Test $TEST: incorrect output from cmsRun -c \"${CONFIG_INPUT}\" ${LOCAL_TEST_DIR}/test_argparse.py"
