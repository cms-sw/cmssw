#!/bin/bash

# Pass in name and status
function die { echo Failure $1: status $2 ; exit $2 ; }


TEST_DIR=src/FWCore/Framework/test

cmsRun ${TEST_DIR}/test_es_notokenget_cfg.py && die "EventSetup get without token did not fail" 1
echo "EventSetup get without token failed as expected"
