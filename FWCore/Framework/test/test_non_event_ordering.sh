#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=$CMSSW_BASE/src/FWCore/Framework/test

echo test_non_event_ordering_beginLumi_cfg.py
cmsRun $TEST_DIR/test_non_event_ordering_beginLumi_cfg.py || die "begin Lumi test failed" $?
echo test_non_event_ordering_beginRun_cfg.py
cmsRun $TEST_DIR/test_non_event_ordering_beginRun_cfg.py || die "begin Run test failed" $?
echo test_non_event_ordering_endLumi_cfg.py
cmsRun $TEST_DIR/test_non_event_ordering_endLumi_cfg.py  || die "end Lumi test failed" $?
echo test_non_event_ordering_endRun_cfg.py
cmsRun $TEST_DIR/test_non_event_ordering_endRun_cfg.py || die "end Run test failed" $?
echo test_non_event_ordering_beginProcessBlock_cfg.py
cmsRun $TEST_DIR/test_non_event_ordering_beginProcessBlock_cfg.py || die "begin ProcessBlock test failed" $?
echo test_non_event_ordering_endProcessBlock_cfg.py
cmsRun $TEST_DIR/test_non_event_ordering_endProcessBlock_cfg.py || die "end Process block test failed" $?
echo test_non_event_ordering_multithreaded_cfg.py
cmsRun $TEST_DIR/test_non_event_ordering_multithreaded_cfg.py || die "multithreaded test failed" $?
