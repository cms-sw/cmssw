#!/bin/bash

# Pass in name and status
function die {
  echo $1: status $2
  echo === Log file ===
  cat ${3:-/dev/null}
  echo === End log file ===
  exit $2
}

# run test job
TESTDIR="${LOCALTOP}"/src/HLTrigger/HLTfilters/test

cmsRun "${TESTDIR}"/triggerResultsFilter_by_PathStatus.py &> log_triggerResultsFilter_by_PathStatus \
 || die "Failure running triggerResultsFilter_by_PathStatus.py" $? log_triggerResultsFilter_by_PathStatus

# expected PathSummary of test job
cat <<@EOF > log_triggerResultsFilter_by_PathStatus_expected
TrigReport ---------- Event  Summary ------------
TrigReport Events total = 1000 passed = 1000 failed = 0
TrigReport ---------- Path   Summary ------------
TrigReport  Trig Bit#   Executed     Passed     Failed      Error Name
TrigReport     1    0       1000        500        500          0 Path_1
TrigReport     1    1       1000        333        667          0 Path_2
TrigReport     1    2       1000        200        800          0 Path_3
TrigReport     1    3       1000       1000          0          0 L1_Path
TrigReport     1    4       1000       1000          0          0 AlwaysNOTFalse
TrigReport     1    5       1000          0       1000          0 AlwaysFALSE
TrigReport     1    6       1000        500        500          0 Check_1
TrigReport     1    7       1000        333        667          0 Check_2
TrigReport     1    8       1000        200        800          0 Check_3
TrigReport     1    9       1000         33        967          0 Check_All_Explicit
TrigReport     1   10       1000        733        267          0 Check_Any_Or
TrigReport     1   11       1000        733        267          0 Check_Any_Star
TrigReport     1   12       1000         33        967          0 Check_1_Pre
TrigReport     1   13       1000        967         33          0 Check_NOT_1_Pre
TrigReport     1   14       1000         33        967          0 Check_2_Pre
TrigReport     1   15       1000         99        901          0 Check_Any_Pre
TrigReport     1   16       1000         99        901          0 Check_Any_Pre_DoubleNOT
TrigReport     1   17       1000        901         99          0 Check_Not_Any_Pre
TrigReport     1   18       1000        733        267          0 Check_Any_Question
TrigReport     1   19       1000        733        267          0 Check_Any_StarQuestion
TrigReport     1   20       1000          0       1000          0 Check_Wrong_Name
TrigReport     1   21       1000          0       1000          0 Check_Wrong_Pattern
TrigReport     1   22       1000       1000          0          0 Check_Not_Wrong_Pattern
TrigReport     1   23       1000          0       1000          0 Check_Empty_Pattern
TrigReport     1   24       1000          0       1000          0 Check_L1Path_Pattern
TrigReport     1   25       1000          0       1000          0 Check_L1Singlemuopen_Pattern
TrigReport     1   26       1000       1000          0          0 Check_True_Pattern
TrigReport     1   27       1000          0       1000          0 Check_False_Pattern
TrigReport     1   28       1000       1000          0          0 Check_AlwaysNOTFalse_Pattern
TrigReport     1   29       1000       1000          0          0 Check_NOTAlwaysFALSE_Pattern
@EOF

# compare to expected output of test job
grep -m$(cat log_triggerResultsFilter_by_PathStatus_expected | wc -l) \
 'TrigReport ' log_triggerResultsFilter_by_PathStatus | diff log_triggerResultsFilter_by_PathStatus_expected - \
 || die "differences in expected log report" $?
