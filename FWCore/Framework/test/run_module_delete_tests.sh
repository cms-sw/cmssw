#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=src/FWCore/Framework/test

cmsRun $TEST_DIR/test_module_delete_cfg.py || die "module deletion test failed" $?
echo "module deletion test succeeded"
cmsRun $TEST_DIR/test_module_delete_subprocess_cfg.py || die "module deletion test with subprocess failed" $?
echo "module deletion test with subprocess succeeded"
cmsRun $TEST_DIR/test_module_delete_improperDependencies_cfg.py && die "module deletion with improper module ordering test failed" $?
echo "module deletion test with improper module ordering succeeded"
cmsRun $TEST_DIR/test_module_delete_looper_cfg.py || die "module deletetion test with looper failed" $?
echo "module deletion test with looper succeeded"
