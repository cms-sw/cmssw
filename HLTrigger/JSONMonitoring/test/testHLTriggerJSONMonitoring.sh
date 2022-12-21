#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  if [ $# -gt 2 ]; then
    printf "%s\n" "=== Log File =========="
    cat $3
    printf "%s\n" "=== End of Log File ==="
  fi
  exit $2
}

# run test job
TESTDIR="${LOCALTOP}"/src/HLTrigger/JSONMonitoring/test

cmsRun "${TESTDIR}"/testHLTriggerJSONMonitoring_cfg.py &> log_testHLTriggerJSONMonitoring \
 || die "Failure using testHLTriggerJSONMonitoring_cfg.py" $? log_testHLTriggerJSONMonitoring

cat log_testHLTriggerJSONMonitoring

# expected PathSummary of test job
cat <<@EOF > log_testHLTriggerJSONMonitoring_expected
TrigReport ---------- Event  Summary ------------
TrigReport Events total = 100 passed = 100 failed = 0
TrigReport ---------- Path   Summary ------------
TrigReport  Trig Bit#   Executed     Passed     Failed      Error Name
TrigReport     1    0        100          0        100          0 HLTriggerFirstPath
TrigReport     1    1        100        100          0          0 HLT_TestPathA_v1
TrigReport     1    2        100         50         50          0 HLT_TestPathB_v1
TrigReport     1    3        100        100          0          0 HLT_TestPathC_v1
TrigReport     1    4        100          0        100          0 HLTriggerFinalPath
TrigReport     1    5        100         33         67          0 Dataset_TestDatasetX
TrigReport     1    6        100          2         98          0 Dataset_TestDatasetY
@EOF

# compare to expected output of test job
grep -m11 TrigReport log_testHLTriggerJSONMonitoring | diff log_testHLTriggerJSONMonitoring_expected - \
  || die "Unexpected differences in outputs of testHLTriggerJSONMonitoring_cfg.py" $?
