#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

TEST_DIR=$CMSSW_BASE/src/FWCore/Framework/test

cmsRun $TEST_DIR/test_module_delete_cfg.py || die "module deletion test failed" $?
echo "module deletion test succeeded"
cmsRun $TEST_DIR/test_module_delete_improperDependencies_cfg.py && die "module deletion with improper module ordering test failed" 1
echo "module deletion test with improper module ordering succeeded"
cmsRun $TEST_DIR/test_module_delete_looper_cfg.py || die "module deletetion test with looper failed" $?
echo "module deletion test with looper succeeded"
cmsRun $TEST_DIR/test_module_delete_dependencygraph_cfg.py || die "module deletetion test with DependencyGraph failed" $?
echo "module deletion test with DependencyGraph succeeded"
cmsRun $TEST_DIR/test_module_delete_disable_cfg.py || die "module deletetion test with disabling the deletion failed" $?
echo "module deletion test with disabling the deletion succeeded"

cmsRun $TEST_DIR/test_module_delete_wantSummary_cfg.py > test_module_delete_wantSummary.log 2>&1 || die "module deletetion test with wantSummary failed" $?
# All expected TrigReport/TimeReport lines plus 1 empty line to catch there are unexpected lines
N_TRIG=$(grep -A 6 "TrigReport.*Module Summary" test_module_delete_wantSummary.log | grep -c "TrigReport")
if [ $N_TRIG -ne 6 ]; then
    echo "module deletetion test with wantSummary failed, expected 6 TrigReport lines, got $N_TRIG"
    cat test_module_delete_wantSummary.log
    exit 1
fi
N_TIME=$(grep -A 6 "TimeReport.*Module Summary" test_module_delete_wantSummary.log | grep -c "TimeReport")
if [ $N_TIME -ne 6 ]; then
    echo "module deletetion test with wantSummary failed, expected 6 TimeReport lines, got $N_TIME"
    cat test_module_delete_wantSummary.log
    exit 1
fi

echo "module deletion test with wantSummary succeeded"
