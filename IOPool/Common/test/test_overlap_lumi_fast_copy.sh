#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${CMSSW_BASE}/src/IOPool/Common/test

cmsRun ${LOCAL_TEST_DIR}/make_overlap_lumi_cfg.py || die "cmsRun make_overlap_lumi_cfg.py failed" $?

cmsRun -e -j overlap_lumi_FrameworkJobReport.xml ${LOCAL_TEST_DIR}/copy_overlap_lumi_cfg.py || die "cmsRun copy_overlap_lumi_cfg.py failed" $?

grep '<FastCopying>1</FastCopying>' overlap_lumi_FrameworkJobReport.xml || die "fast copying did not occur" 1
