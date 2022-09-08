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

cmsRun "${TESTDIR}"/triggerResultsFilter_producer.py &> log_triggerResultsFilter_producer \
  || die "Failure running triggerResultsFilter_producer.py" $? log_triggerResultsFilter_producer

cmsRun "${TESTDIR}"/triggerResultsFilter_by_TriggerResults.py &> log_triggerResultsFilter_by_TriggerResults \
 || die "Failure running triggerResultsFilter_by_TriggerResults.py" $? log_triggerResultsFilter_by_TriggerResults

# expected PathSummary of test job
cat <<@EOF > log_triggerResultsFilter_by_TriggerResults_expected
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
TrigReport     1   14       1000       1000          0          0 path_any_doublestar
TrigReport     1   15       1000        733        267          0 path_any_question
TrigReport     1   16       1000          0       1000          0 path_wrong_name
TrigReport     1   17       1000          0       1000          0 path_wrong_pattern
TrigReport     1   18       1000       1000          0          0 path_not_wrong_pattern
TrigReport     1   19       1000          0       1000          0 path_empty_pattern
TrigReport     1   20       1000          0       1000          0 path_l1path_pattern
TrigReport     1   21       1000          0       1000          0 path_l1singlemuopen_pattern
TrigReport     1   22       1000       1000          0          0 path_true_pattern
TrigReport     1   23       1000          0       1000          0 path_false_pattern
@EOF

# compare to expected output of test job
grep -m$(cat log_triggerResultsFilter_by_TriggerResults_expected | wc -l) \
 'TrigReport ' log_triggerResultsFilter_by_TriggerResults | diff log_triggerResultsFilter_by_TriggerResults_expected - \
 || die "differences in expected log report" $?
