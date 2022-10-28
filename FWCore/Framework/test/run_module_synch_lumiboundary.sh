#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=src/FWCore/Framework/test

cmsRun $TEST_DIR/test_module_synch_lumiboundary_cfg.py && die "module requiring synch on lumi boundaries test failed" 1
echo "module requiring synch on lumi boundaries test failed"
