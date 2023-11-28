#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING AlignmentStats ..."
cmsRun ${CMSSW_BASE}/src/Alignment/CommonAlignmentMonitor/test/testAlignmentStats_cfg.py || die "Failure running testAlignmentStats_cfg.py" $?
