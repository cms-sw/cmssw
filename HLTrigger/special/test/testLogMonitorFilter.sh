#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ; echo === Log file === ; cat ${3:-/dev/null} ; echo === End log file === ; exit $2; }


TESTDIR=${LOCALTOP}/src/HLTrigger/special/test

cmsRun ${TESTDIR}/testLogMonitorFilter.py >& log_log_monitor_filter || die "Failure using testLogMonitorFilter.py" $? log_log_monitor_filter

cat <<EOF  > expected_log_report
Log-Report ---------- HLTLogMonitorFilter Summary ------------
Log-Report  Threshold   Prescale     Issued   Accepted   Rejected Category
Log-Report         10        100       1000         28        972 Other
Log-Report          0          1       1000          0       1000 Test
Log-Report          1          1       1000       1000          0 TestError
Log-Report         20       8000      10000         58       9942 TestWarning
EOF

grep 'Log-Report ' log_log_monitor_filter | diff expected_log_report - || die "differences in expected log report" $?

