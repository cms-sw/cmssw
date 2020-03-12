#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=src/FWCore/Framework/test

cmsRun $TEST_DIR/test_non_event_ordering_beginLumi_cfg.py || die "begin Lumi test failed" $?
cmsRun $TEST_DIR/test_non_event_ordering_beginRun_cfg.py || die "begin Run test failed" $?
cmsRun $TEST_DIR/test_non_event_ordering_endLumi_cfg.py  || die "end Lumi test failed" $?
cmsRun $TEST_DIR/test_non_event_ordering_endRun_cfg.py || die "end Run test failed" $?
