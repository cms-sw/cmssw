#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=src/FWCore/Framework/test

cmsRun $TEST_DIR/test_module_delete_cfg.py || die "module deletion test failed" $?
echo "module deletion test succeeded"
