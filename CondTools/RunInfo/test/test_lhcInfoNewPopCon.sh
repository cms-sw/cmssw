#!/bin/sh

SCRIPTS_DIR=${CMSSW_BASE}/src/CondTools/RunInfo/python

function die { echo Failure $1: status $2 ; exit $2 ; }

assert_equal() {
  expected="$1"
  actual="$2"
  message="$3"
  
  if [ "$expected" != "$actual" ]; then
    die "$message: Expected $expected, but got $actual" 1
  fi
}

function assert_found_fills {
  log_file="$1"
  script_name="$2"
  shift 2
  for fill_nr in "$@"; do
    if ! grep -q "Found fill $fill_nr" "$log_file"; then
      die "$script_name didn't find fill $fill_nr" 1 # TODO FIX
    fi
  done
}

rm -f lhcinfo_pop_unit_test.db

echo "testing LHCInfoPerFillPopConAnalyzer in EndFill mode for startTime=\"2022-10-24 01:00:00.000\" endTime=\"2022-10-24 20:00:00.000\"" 
cmsRun ${SCRIPTS_DIR}/LHCInfoPerFillPopConAnalyzer.py mode=endFill \
    destinationConnection="sqlite_file:lhcinfo_pop_unit_test.db" \
    startTime="2022-10-24 01:00:00.000" endTime="2022-10-24 20:00:00.000" \
    tag=fill_end_test > fill_end_test.log || die "cmsRun LHCInfoPerFillPopConAnalyzer.py mode=endFill" $?
assert_equal 7 `cat fill_end_test.log | grep -E '^Since ' | \
    wc -l` "LHCInfoPerFillPopConAnalyzer in EndFill mode written wrong number of payloads"
assert_found_fills fill_end_test.log "LHCInfoPerFillPopConAnalyzer in EndFill mode" 8307 8309

echo "testing LHCInfoPerLSPopConAnalyzer in endFill mode for startTime=\"2022-10-24 01:00:00.000\" endTime=\"2022-10-24 20:00:00.000\"" 
cmsRun ${SCRIPTS_DIR}/LHCInfoPerLSPopConAnalyzer.py mode=endFill \
    destinationConnection="sqlite_file:lhcinfo_pop_unit_test.db" \
    startTime="2022-10-24 01:00:00.000" endTime="2022-10-24 20:00:00.000" \
    tag=ls_end_test > ls_end_test.log || die "cmsRun LHCInfoPerLSPopConAnalyzer.py mode=endFill" $?
assert_equal 169 `cat ls_end_test.log | grep -E '^Since ' | \
    wc -l` "LHCInfoPerLSPopConAnalyzer in endFill mode written wrong number of payloads"
assert_found_fills ls_end_test.log "LHCInfoPerLSPopConAnalyzer in endFill mode" 8307 8309

echo "testing LHCInfoPerLSPopConAnalyzer in endFill mode for startTime=\"2022-07-11 22:00:00.000\" endTime=\"2022-07-12 18:10:10.000\"" 
cmsRun ${SCRIPTS_DIR}/LHCInfoPerLSPopConAnalyzer.py mode=endFill \
    destinationConnection="sqlite_file:lhcinfo_pop_unit_test.db" \
    startTime="2022-07-11 22:00:00.000" endTime="2022-07-12 18:10:10.000" \
    tag=ls_end_test2 > ls_end_test2.log || die "cmsRun LHCInfoPerLSPopConAnalyzer.py mode=endFill" $?
assert_equal 69 `cat ls_end_test2.log | grep -E '^Since ' | \
    wc -l` "LHCInfoPerLSPopConAnalyzer in endFill mode written wrong number of payloads"
assert_found_fills ls_end_test2.log "LHCInfoPerLSPopConAnalyzer in endFill mode" 7967

echo "testing LHCInfoPerFillPopConAnalyzer in DuringFill mode for startTime=\"2022-10-24 01:00:00.000\" endTime=\"2022-10-24 20:00:00.000\"" 
cmsRun ${SCRIPTS_DIR}/LHCInfoPerFillPopConAnalyzer.py mode=duringFill \
    destinationConnection="sqlite_file:lhcinfo_pop_unit_test.db" \
    startTime="2022-10-24 01:00:00.000" endTime="2022-10-24 20:00:00.000" \
    tag=fill_during_test > fill_during_test.log || die "cmsRun LHCInfoPerFillPopConAnalyzer.py" $?
assert_equal 1 `cat fill_during_test.log | grep -E '^Since ' | \
    wc -l` "LHCInfoPerFillPopConAnalyzer in DuringFill written wrong number of payloads"
assert_found_fills fill_during_test.log "LHCInfoPerFillPopConAnalyzer in DuringFill" 8307 8309

echo "testing LHCInfoPerLSPopConAnalyzer in duringFill mode for startTime=\"2022-10-24 01:00:00.000\" endTime=\"2022-10-24 20:00:00.000\"" 
cmsRun ${SCRIPTS_DIR}/LHCInfoPerLSPopConAnalyzer.py mode=duringFill \
    destinationConnection="sqlite_file:lhcinfo_pop_unit_test.db" \
    startTime="2022-10-24 01:00:00.000" endTime="2022-10-24 20:00:00.000" \
    tag=ls_during_test > ls_during_test.log || die "cmsRun LHCInfoPerLSPopConAnalyzer.py mode=duringFill" $?
assert_equal 1 `cat ls_during_test.log | grep -E '^Since ' | \
    wc -l` "LHCInfoPerLSPopConAnalyzer in duringFill mode written wrong number of payloads"
assert_found_fills ls_during_test.log  "LHCInfoPerLSPopConAnalyzer in duringFill mode" 8307 8309