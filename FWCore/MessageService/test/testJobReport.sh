#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

cmsRun -j fwjr.xml ${SCRAM_TEST_PATH}/jobreport_cfg.py > log.txt 2>&1 || die "cmsRun jobreport_cfg.py failed" $?
${SCRAM_TEST_PATH}/compareJobReportToOutput.py fwjr.xml log.txt || die "Job report comparison to log failed" $?

               
