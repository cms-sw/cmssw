#!/bin/sh

LOCAL_TEST_DIR="${CMSSW_BASE}/src/CondTools/RunInfo/test"
LOG_DIR="test_printLHCInfoPerPayloads_logs"

# Logs and temporary files cleanup
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}"/*.log
trap "rm -f endFill_iovs.txt" EXIT

# Source shared utility functions
source "${LOCAL_TEST_DIR}/testing_utils.sh"

# Create test IOVs file for endFill test
cat > "endFill_iovs.txt" << EOF
7113674196864991232
7113765039718268928
7113769433469812736
7115287966401953792
7115357364483522560
7115389628277850112
EOF

# Test 1: Print PerLS duringFill (lumiid IOVs)
echo "Test 1: Print PerLS duringFill (lumiid IOVs)..."
echo "1686633657139272 1686676606812225 1686771096092709 1686852700471354" | \
${LOCAL_TEST_DIR}/printLHCInfoPerPayloads.py \
    record=LHCInfoPerLS tag=LHCInfoPerLS_duringFill_hlt_v1 timetype=lumiid \
    csv=True header=True > "${LOG_DIR}/perls_duringfill_lumiid.log" 2>&1

lines=$(grep -cve '^\s*$' "${LOG_DIR}/perls_duringfill_lumiid.log")
# Expected: CSV header + 4 data lines = 5 lines
assert_equal 5 "$lines" "PerLS duringFill lumiid test has wrong number of lines" "${LOG_DIR}/perls_duringfill_lumiid.log"

# Test 2: Print PerLS duringFill (lumiid IOVs) with csv=False and header=False
echo "Test 2: Print PerLS duringFill (lumiid IOVs) - no CSV, no header..."
echo "1686633657139272 1686676606812225 1686771096092709 1686852700471354" | \
${LOCAL_TEST_DIR}/printLHCInfoPerPayloads.py \
    record=LHCInfoPerLS tag=LHCInfoPerLS_duringFill_hlt_v1 timetype=lumiid \
    csv=False header=False > "${LOG_DIR}/perls_duringfill_no_csv.log" 2>&1

lines=$(grep -cve '^\s*$' "${LOG_DIR}/perls_duringfill_no_csv.log")
# Expected: 4 IOVs × multiple lines (7) per payload (depends on payload content)
assert_equal 28 "$lines" "PerLS duringFill no CSV test has wrong number of lines" "${LOG_DIR}/perls_duringfill_no_csv.log"

# Test 3: Print PerFill duringFill (lumiid IOVs)
echo "Test 3: Print PerFill duringFill (lumiid IOVs)..."
echo "1686496218185804" | \
${LOCAL_TEST_DIR}/printLHCInfoPerPayloads.py \
    record=LHCInfoPerFill tag=LHCInfoPerFill_duringFill_hlt_v1 \
    timetype=lumiid > "${LOG_DIR}/perfill_duringfill_lumiid.log" 2>&1

lines=$(grep -cve '^\s*$' "${LOG_DIR}/perfill_duringfill_lumiid.log")
# Expected: Multiple lines of payload information (exact count depends on payload content)
assert_equal 31 "$lines" "PerFill duringFill lumiid test has wrong number of lines" "${LOG_DIR}/perfill_duringfill_lumiid.log"

# Test 4: Print PerFill duringFill (lumiid IOVs), filter fill number and energy only
echo "Test 4: Print PerFill duringFill - filtered (lumiid IOVs)..."
echo "1686513398054988 1686633657139272 1686676606812225 1686771096092709 1686852700471354" | \
${LOCAL_TEST_DIR}/printLHCInfoPerPayloads.py \
    record=LHCInfoPerFill tag=LHCInfoPerFill_duringFill_hlt_v1 timetype=lumiid 2>/dev/null | \

grep -E "LHC fill|Energy" > "${LOG_DIR}/perfill_duringfill_filtered.log" 2>&1

lines=$(grep -cve '^\s*$' "${LOG_DIR}/perfill_duringfill_filtered.log")
# Expected: 2 lines per IOV (LHC fill + Energy) × 5 IOVs = 10 lines
assert_equal 10 "$lines" "PerFill duringFill filtered test has wrong number of lines" "${LOG_DIR}/perfill_duringfill_filtered.log"

# Test 5: Print PerFill endFill (timestamp IOVs), filter fill number, energy and fill creation time only
echo "Test 5: Print PerFill endFill - filtered (timestamp IOVs from file)..."
${LOCAL_TEST_DIR}/printLHCInfoPerPayloads.py \
    record=LHCInfoPerFill tag=LHCInfoPerFill_endFill_Run3_v2 timetype=timestamp < "endFill_iovs.txt" 2>/dev/null | \
    
grep -E "Energy|LHC fill|Creation time" > "${LOG_DIR}/perfill_endfill_filtered.log" 2>&1

lines=$(grep -cve '^\s*$' "${LOG_DIR}/perfill_endfill_filtered.log")
# Expected: 3 lines per IOV × 6 IOVs = 18 lines
assert_equal 18 "$lines" "PerFill endFill filtered test has wrong number of lines" "${LOG_DIR}/perfill_endfill_filtered.log"

echo "All payload printing tests completed successfully!"
echo "Logs with tests output are stored in: ${LOG_DIR}/"