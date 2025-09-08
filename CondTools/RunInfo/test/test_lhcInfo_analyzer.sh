#!/bin/sh

LOCAL_TEST_DIR=$CMSSW_BASE/src/CondTools/RunInfo/test
LOG_DIR="test_lhcInfo_analyzer_logs"

# Create log directory and clean existing logs
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}"/*.log
rm -f LHCInfoPerFill.sqlite LHCInfoPerLS.sqlite

# Source shared utility functions
source "${LOCAL_TEST_DIR}/testing_utils.sh"

# Run LHCInfoPerFill writer
echo "Running LHCInfoPerFill writer..."
cmsRun "${LOCAL_TEST_DIR}/LHCInfoPerFillWriter_cfg.py" || die "cmsRun LHCInfoPerFillWriter_cfg.py" $?

# LHCInfoPerFill format test case
echo "Running LHCInfoPerFill analyzer..."
cmsRun "${LOCAL_TEST_DIR}/LHCInfoPerFillAnalyzer_cfg.py" \
    tag=LHCInfoPerFillFake \
    db=sqlite_file:LHCInfoPerFill.sqlite > "${LOG_DIR}/fill_analyzer.log" 2>&1 \
    || die "cmsRun LHCInfoPerFillAnalyzer_cfg.py" $?

lines=$(grep -cve '^\s*$' "${LOG_DIR}/fill_analyzer.log")
# Expected: 31 lines (IOV print, 'LHCInfoPerFill retrieved' + 29 payload field lines)
assert_equal 31 "$lines" "LHCInfoPerFillAnalyzer_cfg.py log has wrong number of lines" "${LOG_DIR}/fill_analyzer.log"

# Run LHCInfoPerLS writer
echo "Running LHCInfoPerLS writer..."
cmsRun "${LOCAL_TEST_DIR}/LHCInfoPerLSWriter_cfg.py" || die "cmsRun LHCInfoPerLSWriter_cfg.py" $?

# LHCInfoPerLS format csv=True test case
echo "Running LHCInfoPerLS analyzer (CSV format)..."
cmsRun "${LOCAL_TEST_DIR}/LHCInfoPerLSAnalyzer_cfg.py" \
    tag=LHCInfoPerLSFake \
    db=sqlite_file:LHCInfoPerLS.sqlite \
    csv=True \
    header=True > "${LOG_DIR}/ls_analyzer_csv.log" 2>&1 \
    || die "cmsRun LHCInfoPerLSAnalyzer_cfg.py" $?

lines=$(grep -cve '^\s*$' "${LOG_DIR}/ls_analyzer_csv.log")
# Expected: 2 lines due to CSV format
assert_equal 2 "$lines" "LHCInfoPerLSAnalyzer_cfg.py log has wrong number of lines" "${LOG_DIR}/ls_analyzer_csv.log"

# LHCInfoPerLS format csv=False test case
echo "Running LHCInfoPerLS analyzer (non-CSV format)..."
cmsRun "${LOCAL_TEST_DIR}/LHCInfoPerLSAnalyzer_cfg.py" \
    tag=LHCInfoPerLSFake \
    db=sqlite_file:LHCInfoPerLS.sqlite \
    csv=False > "${LOG_DIR}/ls_analyzer_no_csv.log" 2>&1 \
    || die "cmsRun LHCInfoPerLSAnalyzer_cfg.py" $?

lines=$(grep -cve '^\s*$' "${LOG_DIR}/ls_analyzer_no_csv.log")
# Expected: 7 lines (variables returned in LHCInfoPerLSAnalyzer)
assert_equal 7 "$lines" "LHCInfoPerLSAnalyzer_cfg.py log has wrong number of lines" "${LOG_DIR}/ls_analyzer_no_csv.log"

echo "All tests completed successfully!"
echo "Logs including test output are stored in: ${LOG_DIR}/"