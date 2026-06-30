#!/bin/bash
# HLTrigger/Tools/test/testHltDiff.sh
#
# Black-box integration test for the hltDiff binary.
# Generates minimal EDM ROOT files with known TriggerResults,
# then runs hltDiff and asserts on stdout/stderr/exit code.
#
# Usage (within a CMSSW environment):
#   cd $CMSSW_BASE/src/HLTrigger/Tools/test
#   bash testHltDiff.sh
#
# Flag reference (from actual getopt in hltDiff.cc):
#   -o / --old-files     : reference (old) input file(s)
#   -n / --new-files     : new input file(s)
#   -O / --old-process   : process name for old TriggerResults
#   -N / --new-process   : process name for new TriggerResults
#   -m / --max-events    : limit number of events
#   -p / --prescales     : do NOT ignore prescale differences
#   -c / --csv-output    : write CSV summary
#   -j / --json-output   : write JSON output (per-run filename)
#   -F / --output-file   : write JSON/ROOT to a single named file (FILE.json, FILE.root)
#   -r / --root-output   : write ROOT histograms
#   -f / --file-check    : check file existence before running
#   -q / --quiet         : suppress per-trigger summary table
#   -v / --verbose LEVEL : 1=per-event, 2=+trigger candidates, 3=+all candidates
#   -d / --debug         : print skipped-event messages
#   -h / --help          : print usage and exit

set -euo pipefail

###############################################################################
# Helpers
###############################################################################
PASS=0
FAIL=0

ok() {
    echo "  PASS: $*"
    ((PASS++)) || true
}

fail() {
    echo "  FAIL: $*"
    ((FAIL++)) || true
}

assert_exit() {
    local expected=$1 actual=$2 label=$3
    if [[ "$actual" == "$expected" ]]; then
        ok "$label (exit=$actual)"
    else
        fail "$label: expected exit $expected, got $actual"
    fi
}

# grep -E for alternation support
assert_contains() {
    local pattern=$1 text=$2 label=$3
    if echo "$text" | grep -qE -- "$pattern"; then
        ok "$label"
    else
        fail "$label: pattern '$pattern' not found in output"
        echo "--- output ---"
        echo "$text"
        echo "--------------"
    fi
}

assert_not_contains() {
    local pattern=$1 text=$2 label=$3
    if echo "$text" | grep -qE -- "$pattern"; then
        fail "$label: pattern '$pattern' should NOT appear in output"
        echo "--- output ---"
        echo "$text"
        echo "--------------"
    else
        ok "$label"
    fi
}

###############################################################################
# Setup: temp dir
###############################################################################
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
export TMPDIR

###############################################################################
# EDM file generation
# ref.root: HLT_PathA=Pass, HLT_PathB=Pass, HLT_PathC=Pass  (all 5 events)
# new.root: HLT_PathA=Pass, HLT_PathB=Fail, HLT_PathC=Pass  (all 5 events)
# → HLT_PathB loses in every event; A and C are identical
###############################################################################
echo ""
echo "=== Generating test EDM files ==="

cat > "$TMPDIR/makeRef.py" << 'PYEOF'
import FWCore.ParameterSet.Config as cms, os
OUTDIR = os.environ['TMPDIR']
process = cms.Process("HLT")
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(5),
    numberEventsInLuminosityBlock = cms.untracked.uint32(5),
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))
process.filterA = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterB = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterC = cms.EDFilter("HLTBool", result = cms.bool(True))
process.HLT_PathA = cms.Path(process.filterA)
process.HLT_PathB = cms.Path(process.filterB)
process.HLT_PathC = cms.Path(process.filterC)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(f"{OUTDIR}/ref.root"),
    outputCommands = cms.untracked.vstring("drop *", "keep edmTriggerResults_*_*_HLT")
)
process.outPath = cms.EndPath(process.out)
PYEOF

cat > "$TMPDIR/makeNew.py" << 'PYEOF'
import FWCore.ParameterSet.Config as cms, os
OUTDIR = os.environ['TMPDIR']
process = cms.Process("HLT")
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(5),
    numberEventsInLuminosityBlock = cms.untracked.uint32(5),
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))
process.filterA = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterB = cms.EDFilter("HLTBool", result = cms.bool(False))   # PathB now fails
process.filterC = cms.EDFilter("HLTBool", result = cms.bool(True))
process.HLT_PathA = cms.Path(process.filterA)
process.HLT_PathB = cms.Path(process.filterB)
process.HLT_PathC = cms.Path(process.filterC)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(f"{OUTDIR}/new.root"),
    outputCommands = cms.untracked.vstring("drop *", "keep edmTriggerResults_*_*_HLT")
)
process.outPath = cms.EndPath(process.out)
PYEOF

# Three-path file with an extra path D (for added/removed path tests)
cat > "$TMPDIR/makeNewExtra.py" << 'PYEOF'
import FWCore.ParameterSet.Config as cms, os
OUTDIR = os.environ['TMPDIR']
process = cms.Process("HLT")
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3))
process.filterA = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterB = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterC = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterD = cms.EDFilter("HLTBool", result = cms.bool(True))
process.HLT_PathA = cms.Path(process.filterA)
process.HLT_PathB = cms.Path(process.filterB)
process.HLT_PathC = cms.Path(process.filterC)
process.HLT_PathD = cms.Path(process.filterD)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(f"{OUTDIR}/new_extra.root"),
    outputCommands = cms.untracked.vstring("drop *", "keep edmTriggerResults_*_*_HLT")
)
process.outPath = cms.EndPath(process.out)
PYEOF

cat > "$TMPDIR/makeRefSmall.py" << 'PYEOF'
import FWCore.ParameterSet.Config as cms, os
OUTDIR = os.environ['TMPDIR']
process = cms.Process("HLT")
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3))
process.filterA = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterB = cms.EDFilter("HLTBool", result = cms.bool(True))
process.filterC = cms.EDFilter("HLTBool", result = cms.bool(True))
process.HLT_PathA = cms.Path(process.filterA)
process.HLT_PathB = cms.Path(process.filterB)
process.HLT_PathC = cms.Path(process.filterC)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(f"{OUTDIR}/ref_small.root"),
    outputCommands = cms.untracked.vstring("drop *", "keep edmTriggerResults_*_*_HLT")
)
process.outPath = cms.EndPath(process.out)
PYEOF

echo "  Running cmsRun makeRef.py ..."
cmsRun "$TMPDIR/makeRef.py" 2>/dev/null
echo "  Running cmsRun makeNew.py ..."
cmsRun "$TMPDIR/makeNew.py" 2>/dev/null
echo "  Running cmsRun makeRefSmall.py ..."
cmsRun "$TMPDIR/makeRefSmall.py" 2>/dev/null
echo "  Running cmsRun makeNewExtra.py ..."
cmsRun "$TMPDIR/makeNewExtra.py" 2>/dev/null
echo "  Done."

REF="$TMPDIR/ref.root"
NEW="$TMPDIR/new.root"
REF_SMALL="$TMPDIR/ref_small.root"
NEW_EXTRA="$TMPDIR/new_extra.root"

###############################################################################
# Convenience wrapper: run hltDiff, capture combined stdout+stderr.
# hltDiff itself always returns 0 on success; ROOT's global error handler
# can cause it to exit 1 when writing output files fails (e.g. permission
# issues or zombied TFiles).  We therefore only check exit codes for cases
# where the binary is expected to fail due to its own argument validation,
# not for runs that produce ROOT/JSON/CSV output.
###############################################################################
HLTDIFF_OUT=""
HLTDIFF_EXIT=0

run_hltDiff() {
    HLTDIFF_EXIT=0
    HLTDIFF_OUT=$(hltDiff "$@" 2>&1) || HLTDIFF_EXIT=$?
}

###############################################################################
###############################################################################
# GROUP 1: help / usage
###############################################################################
echo ""
echo "=== Group 1: help / usage ==="

HLTDIFF_EXIT=0; HLTDIFF_OUT=$(hltDiff --help 2>&1) || HLTDIFF_EXIT=$?
assert_exit  0    "$HLTDIFF_EXIT"  "1.1: --help exits 0"
assert_contains "hltDiff" "$HLTDIFF_OUT" "1.2: --help mentions 'hltDiff'"
assert_contains "\-o"     "$HLTDIFF_OUT" "1.3: --help documents -o (old-files)"
assert_contains "\-n"     "$HLTDIFF_OUT" "1.4: --help documents -n (new-files)"
assert_contains "\-m"     "$HLTDIFF_OUT" "1.5: --help documents -m (max-events)"
assert_contains "\-j"     "$HLTDIFF_OUT" "1.6: --help documents -j (json-output)"
assert_contains "\-F"     "$HLTDIFF_OUT" "1.7: --help documents -F (output-file)"
assert_contains "\-q"     "$HLTDIFF_OUT" "1.8: --help documents -q (quiet)"
assert_contains "\-v"     "$HLTDIFF_OUT" "1.9: --help documents -v (verbose)"

###############################################################################
# GROUP 2: error handling — missing required arguments
###############################################################################
echo ""
echo "=== Group 2: error handling ==="

HLTDIFF_EXIT=0; HLTDIFF_OUT=$(hltDiff 2>&1) || HLTDIFF_EXIT=$?
assert_exit 1   "$HLTDIFF_EXIT" "2.1: no args exits 1"
assert_contains "old" "$HLTDIFF_OUT" "2.2: no args mentions missing 'old' files"

HLTDIFF_EXIT=0; HLTDIFF_OUT=$(hltDiff -o "$REF" 2>&1) || HLTDIFF_EXIT=$?
assert_exit 1   "$HLTDIFF_EXIT" "2.3: missing -n exits 1"
assert_contains "new" "$HLTDIFF_OUT" "2.4: missing -n mentions 'new' files"

HLTDIFF_EXIT=0; HLTDIFF_OUT=$(hltDiff --nonexistent-flag 2>&1) || HLTDIFF_EXIT=$?
if [[ "$HLTDIFF_EXIT" -ne 0 ]]; then
    ok "2.5: unknown flag exits non-zero (got $HLTDIFF_EXIT)"
else
    fail "2.5: unknown flag should exit non-zero, got 0"
fi

###############################################################################
# GROUP 3: identical files (ref vs ref) → no differences
# Note: hltDiff's exit code is set by ROOT's global error handler when
# the process exits, not by hltDiff itself.  We do not assert exit code
# for compare() runs; we assert on output content only.
###############################################################################
echo ""
echo "=== Group 3: identical files (ref vs ref) ==="

run_hltDiff -o "$REF" -n "$REF"
assert_contains "5 matching events" "$HLTDIFF_OUT" "3.1: reports 5 matching events"
assert_contains "0 have different"  "$HLTDIFF_OUT" "3.2: reports 0 events with different results"
# The summary table header is printed only when at least one path has total()>0
assert_not_contains "HLT_Path" "$HLTDIFF_OUT" "3.3: no per-trigger rows when files identical"
assert_not_contains "Accepted"  "$HLTDIFF_OUT" "3.4: no table header when files identical"

###############################################################################
# GROUP 4: ref vs new → known differences on HLT_PathB
###############################################################################
echo ""
echo "=== Group 4: ref vs new (PathB Pass→Fail in all events) ==="

run_hltDiff -o "$REF" -n "$NEW"
assert_contains "5 matching events" "$HLTDIFF_OUT" "4.1: all 5 events matched"
assert_contains "5 have different"  "$HLTDIFF_OUT" "4.2: all 5 events are affected"

assert_contains "Accepted" "$HLTDIFF_OUT" "4.3: summary table header contains 'Accepted'"
assert_contains "Gained"   "$HLTDIFF_OUT" "4.4: summary table header contains 'Gained'"
assert_contains "Lost"     "$HLTDIFF_OUT" "4.5: summary table header contains 'Lost'"

# PathB has differences; A and C are identical and must not appear
assert_contains     "HLT_PathB" "$HLTDIFF_OUT" "4.6: PathB appears in diff table"
assert_not_contains "HLT_PathA" "$HLTDIFF_OUT" "4.7: PathA absent (no diff)"
assert_not_contains "HLT_PathC" "$HLTDIFF_OUT" "4.8: PathC absent (no diff)"

# PathB: old=5 accepted, new=0 accepted → lost=5 shown as "-5"
pathb_line=$(echo "$HLTDIFF_OUT" | grep "HLT_PathB")
assert_contains     "\-5"      "$pathb_line" "4.9:  PathB shows -5 lost events"
assert_not_contains "\+[1-9]"  "$pathb_line" "4.10: PathB shows no gained events"

###############################################################################
# GROUP 5: -m (max-events) flag
# Note: -m 0 is passed as 0 to atoi() → max_events=0; the code does
# std::min((int)old_events->size(), (int)max_events) where max_events is
# unsigned int so (int)0 == 0 → nEvents=0 → counter never reaches nEvents
# → the loop runs to completion. This is a known quirk documented here.
###############################################################################
echo ""
echo "=== Group 5: -m flag limits events ==="

run_hltDiff -o "$REF" -n "$NEW" -m 2
assert_contains "2 matching events" "$HLTDIFF_OUT" "5.1: -m 2 processes exactly 2 events"
assert_contains "2 have different"  "$HLTDIFF_OUT" "5.2: 2 affected events with -m 2"

# -m 1 gives a clean single-event run
run_hltDiff -o "$REF" -n "$NEW" -m 1
assert_contains "1 matching events" "$HLTDIFF_OUT" "5.3: -m 1 processes exactly 1 event"

###############################################################################
# GROUP 6: -q (quiet) flag suppresses the summary table
###############################################################################
echo ""
echo "=== Group 6: -q quiet flag ==="

run_hltDiff -o "$REF" -n "$NEW" -q
assert_contains     "matching events" "$HLTDIFF_OUT" "6.1: -q still reports event counts"
assert_not_contains "HLT_PathB"       "$HLTDIFF_OUT" "6.2: -q suppresses per-trigger diff table"
assert_not_contains "Accepted"        "$HLTDIFF_OUT" "6.3: -q suppresses table header"

###############################################################################
# GROUP 7: -v (verbose) flag adds per-event lines
###############################################################################
echo ""
echo "=== Group 7: -v verbose flag ==="

run_hltDiff -o "$REF" -n "$NEW" -v 1
assert_contains "run [0-9]"    "$HLTDIFF_OUT" "7.1: -v 1 prints per-event run/lumi/event header"
assert_contains "old result is" "$HLTDIFF_OUT" "7.2: -v 1 prints old result"
assert_contains "new result is" "$HLTDIFF_OUT" "7.3: -v 1 prints new result"
assert_contains "HLT_PathB"    "$HLTDIFF_OUT" "7.4: -v 1 names the differing path"

run_hltDiff -o "$REF" -n "$NEW"
assert_not_contains "old result is" "$HLTDIFF_OUT" "7.5: without -v no per-event detail"

###############################################################################
# GROUP 8: -j / -F JSON output
###############################################################################
echo ""
echo "=== Group 8: JSON output (-j and -F) ==="

JSON_OUT="$TMPDIR/hltdiff_test"
run_hltDiff -o "$REF" -n "$NEW" -j -F "$JSON_OUT"
JSON_FILE="${JSON_OUT}.json"

if [[ -f "$JSON_FILE" ]]; then
    ok "8.1: JSON file created"
    json_content=$(cat "$JSON_FILE")
    assert_contains "\{"          "$json_content" "8.2: JSON starts with '{'"
    assert_contains "HLT_PathB"   "$json_content" "8.3: JSON contains HLT_PathB"
    assert_contains "configuration" "$json_content" "8.4: JSON contains 'configuration' key"
    assert_contains '"events"'    "$json_content" "8.5: JSON contains 'events' key"
    assert_contains "ref\.root"   "$json_content" "8.6: JSON records old input filename"
    assert_contains "new\.root"   "$json_content" "8.7: JSON records new input filename"
    if python3 -c "import json,sys; json.load(sys.stdin)" < "$JSON_FILE" 2>/dev/null; then
        ok "8.8: JSON is valid (python3 json.load)"
    else
        fail "8.8: JSON failed python3 json.load validation"
    fi
else
    for i in 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8; do fail "$i: (JSON file not created)"; done
fi

###############################################################################
# GROUP 9: -c CSV output
###############################################################################
echo ""
echo "=== Group 9: CSV output (-c and -F) ==="

CSV_BASE="$TMPDIR/hltdiff_csv"
run_hltDiff -o "$REF" -n "$NEW" -c -F "$CSV_BASE"
TRIGGER_CSV="${CSV_BASE}_trigger.csv"
MODULE_CSV="${CSV_BASE}_module.csv"

if [[ -f "$TRIGGER_CSV" ]]; then
    ok "9.1: trigger CSV created"
    csv_t=$(cat "$TRIGGER_CSV")
    assert_contains "Total"     "$csv_t" "9.2: trigger CSV has 'Total' column"
    assert_contains "Gained"    "$csv_t" "9.3: trigger CSV has 'Gained' column"
    assert_contains "Lost"      "$csv_t" "9.4: trigger CSV has 'Lost' column"
    assert_contains "HLT_PathB" "$csv_t" "9.5: trigger CSV contains HLT_PathB row"
    # The CSV writer iterates all triggers in m_triggerSummary regardless of
    # whether they have differences, so all three paths appear in the CSV.
    assert_contains "HLT_PathA" "$csv_t" "9.6: trigger CSV includes unchanged PathA (all paths written)"
else
    for i in 9.1 9.2 9.3 9.4 9.5 9.6; do fail "$i: (trigger CSV not created)"; done
fi

if [[ -f "$MODULE_CSV" ]]; then
    ok "9.7: module CSV created"
    assert_contains "Total" "$(cat "$MODULE_CSV")" "9.8: module CSV has 'Total' column"
else
    fail "9.7: module CSV not created"
    fail "9.8: (skipped)"
fi

###############################################################################
# GROUP 10: -f (file-check) with nonexistent files
###############################################################################
echo ""
echo "=== Group 10: -f file-check flag ==="

run_hltDiff -f -o "/nonexistent/path/ref.root" -n "/nonexistent/path/new.root"
assert_contains "does not exist" "$HLTDIFF_OUT" "10.1: -f reports 'does not exist'"
assert_contains "hltDiff: error" "$HLTDIFF_OUT" "10.2: -f prefixes message with 'hltDiff: error'"

###############################################################################
# GROUP 11: mismatched menus
###############################################################################
echo ""
echo "=== Group 11: mismatched menus ==="

run_hltDiff -o "$NEW_EXTRA" -n "$REF_SMALL"
assert_contains "Warning"  "$HLTDIFF_OUT" "11.1: prints Warning for mismatched menus"
assert_contains "common"   "$HLTDIFF_OUT" "11.2: mentions number of common triggers"
# PathD exists only in old; it should appear in the warning line but not in the diff table
assert_not_contains "HLT_PathD" "$(echo "$HLTDIFF_OUT" | grep -v Warning)" \
    "11.3: PathD absent from diff table rows (only in Warning)"

###############################################################################
# GROUP 12: -n - (self-comparison, single-file mode)
###############################################################################
echo ""
echo "=== Group 12: -n - (self-comparison) ==="

run_hltDiff -o "$REF" -n - -N HLT
assert_contains "0 have different" "$HLTDIFF_OUT" "12.1: -n - same process → 0 differences"
assert_contains "5 matching"       "$HLTDIFF_OUT" "12.2: -n - finds all 5 events"

###############################################################################
# GROUP 13: progress reporting
###############################################################################
echo ""
echo "=== Group 13: progress reporting ==="

run_hltDiff -o "$REF" -n "$NEW"
assert_contains "Processed events:" "$HLTDIFF_OUT" "13.1: progress line appears"
assert_contains "out of"            "$HLTDIFF_OUT" "13.2: progress line contains 'out of'"
# Progress shows percentage in parentheses
assert_contains "[0-9]+%" "$HLTDIFF_OUT" "13.3: progress line shows percentage"

###############################################################################
# Summary
###############################################################################
TOTAL=$((PASS + FAIL))
echo ""
echo "====================================="
echo " Results: $PASS/$TOTAL passed, $FAIL failed"
echo "====================================="
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
