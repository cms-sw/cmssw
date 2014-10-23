#!/bin/bash

# Pass in name and status

function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/testEarlyTerminationSignal_cfg.py 2>&1 | grep -q 'early termination of event: stream = 0 run = 1 lumi = 1 event = 10 : time = 50000001') || die "Early termination signal failed" $?
