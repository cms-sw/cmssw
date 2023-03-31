#!/bin/bash

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

# Pass in name and status
function die { echo $1: status $2 ; echo === Log file === ; cat ${3:-/dev/null} ; echo === End log file === ; exit $2; }

cat <<EOF > finalpath_expected_empty.log
EOF

cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py >& finalpath.log || die "failed test_finalpath_cfg.py" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_empty.log - || die "differences for test_finalpath_cfg.py" $?


cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --schedule >& finalpath.log || die "failed test_finalpath_cfg.py --schedule" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_empty.log - || die "differences for test_finalpath_cfg.py" $?


cat <<EOF > finalpath_expected_not_found.log
did not find thing '' TEST
did not find thing '' TEST
did not find thing '' TEST
found thing 'beginLumi' TEST
found thing 'endLumi' TEST
found thing 'beginRun' TEST
found thing 'endRun' TEST
EOF
cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --schedule --task >& finalpath.log || die "failed test_finalpath_cfg.py --schedule --task" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_not_found.log - || die "differences for test_finalpath_cfg.py --schedule --task" $?


cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --endpath >& finalpath.log || die "failed test_finalpath_cfg.py --endpath" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_empty.log - || die "differences for test_finalpath_cfg.py --endpath" $?

cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --schedule --endpath >& finalpath.log || die "failed test_finalpath_cfg.py --schedule --endpath" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_empty.log - || die "differences for test_finalpath_cfg.py --schedule --endpath" $?


cat <<EOF > finalpath_expected_found.log
found thing '' TEST
found thing '' TEST
found thing '' TEST
found thing 'beginLumi' TEST
found thing 'endLumi' TEST
found thing 'beginRun' TEST
found thing 'endRun' TEST
EOF
cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --endpath --task >& finalpath.log || die "failed test_finalpath_cfg.py --endpath --task" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_found.log - || die "differences for test_finalpath_cfg.py --endpath --task" $?

cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --endpath --task --schedule >& finalpath.log || die "failed test_finalpath_cfg.py --endpath --task --schedule" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_found.log - || die "differences for test_finalpath_cfg.py --endpath --task --schedule" $?

cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --path --task >& finalpath.log || die "failed test_finalpath_cfg.py --path --task" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_found.log - || die "differences for test_finalpath_cfg.py --path --task" $?

cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --path --task --schedule >& finalpath.log || die "failed test_finalpath_cfg.py --path --task --schedule" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_found.log - || die "differences for test_finalpath_cfg.py --path --task --schedule" $?


cat <<EOF > finalpath_expected_filter.log
did not find thing '' TEST
found thing '' TEST
did not find thing '' TEST
found thing 'beginLumi' TEST
found thing 'endLumi' TEST
found thing 'beginRun' TEST
found thing 'endRun' TEST
EOF

cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --path --filter >& finalpath.log || die "failed test_finalpath_cfg.py --path --filter" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_filter.log - || die "differences for test_finalpath_cfg.py --path --filter" $?


cmsRun  ${LOCAL_TEST_DIR}/test_finalpath_cfg.py -- --path --filter --task >& finalpath.log || die "failed test_finalpath_cfg.py --path --filter --task" $?
grep "thing '.*' TEST" finalpath.log | diff finalpath_expected_filter.log - || die "differences for test_finalpath_cfg.py --path --filter --task" $?
