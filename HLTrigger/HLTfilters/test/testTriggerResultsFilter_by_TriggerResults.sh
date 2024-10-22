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
TESTDIR="${LOCALTOP}"/src/HLTrigger/HLTfilters/test

cmsRun "${TESTDIR}"/testTriggerResultsFilter_producer_cfg.py &> log_testTriggerResultsFilter_producer \
  || die "Failure running testTriggerResultsFilter_producer_cfg.py" $? log_testTriggerResultsFilter_producer

cat log_testTriggerResultsFilter_producer

cmsRun "${TESTDIR}"/testTriggerResultsFilter_by_TriggerResults_cfg.py &> log_testTriggerResultsFilter_by_TriggerResults \
  || die "Failure running testTriggerResultsFilter_by_TriggerResults_cfg.py" $? log_testTriggerResultsFilter_by_TriggerResults

cat log_testTriggerResultsFilter_by_TriggerResults

# expected PathSummary of test job
cat <<@EOF > log_testTriggerResultsFilter_by_TriggerResults_expected
TrigReport ---------- Event  Summary ------------
TrigReport Events total = 1000 passed = 1000 failed = 0
TrigReport ---------- Path   Summary ------------
TrigReport  Trig Bit#   Executed     Passed     Failed      Error Name
TrigReport     1    0       1000        500        500          0 path_1
TrigReport     1    1       1000        333        667          0 path_2
TrigReport     1    2       1000        200        800          0 path_3
TrigReport     1    3       1000         33        967          0 path_all_explicit
TrigReport     1    4       1000        733        267          0 path_any_or
TrigReport     1    5       1000       1000          0          0 path_any_star
TrigReport     1    6       1000         33        967          0 path_1_pre
TrigReport     1    7       1000         33        967          0 path_1_pre_with_masks1
TrigReport     1    8       1000         33        967          0 path_1_pre_with_masks2
TrigReport     1    9       1000        967         33          0 path_not_1_pre
TrigReport     1   10       1000         33        967          0 path_2_pre
TrigReport     1   11       1000         99        901          0 path_any_pre
TrigReport     1   12       1000         99        901          0 path_any_pre_doubleNOT
TrigReport     1   13       1000        901         99          0 path_not_any_pre
TrigReport     1   14       1000        499        501          0 Check_1xor2_withoutXOR
TrigReport     1   15       1000        499        501          0 Check_1xor2_withXOR
TrigReport     1   16       1000       1000          0          0 path_any_doublestar
TrigReport     1   17       1000        733        267          0 path_any_question
TrigReport     1   18       1000          0       1000          0 path_wrong_name
TrigReport     1   19       1000          0       1000          0 path_wrong_pattern
TrigReport     1   20       1000       1000          0          0 path_not_wrong_pattern
TrigReport     1   21       1000          0       1000          0 path_empty_pattern
TrigReport     1   22       1000          0       1000          0 path_l1path_pattern
TrigReport     1   23       1000          0       1000          0 path_l1singlemuopen_pattern
TrigReport     1   24       1000       1000          0          0 path_true_pattern
TrigReport     1   25       1000          0       1000          0 path_false_pattern
@EOF

# compare to expected output of test job
grep -m$(cat log_testTriggerResultsFilter_by_TriggerResults_expected | wc -l) \
  'TrigReport ' log_testTriggerResultsFilter_by_TriggerResults | diff log_testTriggerResultsFilter_by_TriggerResults_expected - \
  || die "Unexpected differences in outputs of testTriggerResultsFilter_by_TriggerResults_cfg.py" $?
