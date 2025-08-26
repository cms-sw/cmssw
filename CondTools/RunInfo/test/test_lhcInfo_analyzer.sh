#!/bin/sh

LOCAL_TEST_DIR=$CMSSW_BASE/src/CondTools/RunInfo/test

# Always clean up logs when the script exits (success or failure)
trap "rm -f fill_Analyzer.log ls_Analyzer.log" EXIT

function die { echo Failure $1: status $2 ; exit $2 ; }

assert_equal() {
  expected="$1"
  actual="$2"
  message="$3"
  log_file="$4"

  if [ "$expected" != "$actual" ]; then
    echo "Log output (last 5 lines):"
    tail -n 5 "$log_file"
    die "$message (expected $expected, got $actual)"
  fi
}

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerFillWriter_cfg.py || die "cmsRun LHCInfoPerFillWriter_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerFillAnalyzer_cfg.py \
  tag=LHCInfoPerFillFake \
  db=sqlite_file:LHCInfoPerFill.sqlite > fill_Analyzer.log 2>&1 \
  || die "cmsRun LHCInfoPerFillAnalyzer_cfg.py" $? fill_Analyzer.log
lines=$(wc -l < fill_Analyzer.log)
# Number of lines expected to be 31, accounting IOV print, 'LHCInfoPerFill retrieved' and number of fields
assert_equal 31 "$lines" "LHCInfoPerFillAnalyzer_cfg.py log has wrong number of lines" "fill_Analyzer.log"
rm -f fill_Analyzer.log

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerLSWriter_cfg.py   || die "cmsRun LHCInfoPerLSWriter_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/LHCInfoPerLSAnalyzer_cfg.py \
  tag=LHCInfoPerLSFake \
  db=sqlite_file:LHCInfoPerLS.sqlite > ls_Analyzer.log 2>&1 \
  || die "cmsRun LHCInfoPerLSAnalyzer_cfg.py" $? ls_Analyzer.log
# Number of lines expected to be 31, accounting IOV print, 'LHCInfoPerFill retrieved' and number of fields
assert_equal 31 "$lines" "LHCInfoPerLSAnalyzer_cfg.py log has wrong number of lines" "ls_Analyzer.log"
rm -f ls_Analyzer.log