#!/bin/bash

# Report a failed test, and count it: pass in the message and the status.
# All the tests are run even if some of them fail; the script exits with an error if any of them failed.
FAILED=0
function fail { echo $1: status $2 ; FAILED=$((FAILED + 1)); }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/ParameterSet/test"
REFOUT="${LOCAL_TEST_DIR}/unit_test_outputs/dump.py"

# The pickled configurations are not reproducible byte by byte, so they are compared to the reference by expanding
# them with edmConfigDump.
function checkDump() {
    TEST="$1"
    CFG="$2"
    edmConfigDump -o dump_test$TEST.py $CFG >& dump_test$TEST.log || { fail "Test $TEST: failure running edmConfigDump $CFG" $?; return 1; }
    (diff $REFOUT dump_test$TEST.py) || { fail "Test $TEST: incorrect output from edmConfigDump $CFG" $?; return 1; }
}

# Run edmConfigPickle, writing a self-contained python configuration, and compare it with the reference.
function doTest() {
    TEST="$1"
    CMD="$2"
    OUT="$3"
    LOG="log_test$TEST.log"
    $CMD >& $LOG || { fail "Test $TEST: failure running $CMD" $?; return 1; }
    checkDump $TEST $OUT
}

# Check that a file contains a line matching a regular expression.
function checkLine() {
    TEST="$1"
    FILE="$2"
    LINE="$3"
    grep -q "$LINE" $FILE || { fail "Test $TEST: $FILE does not contain \"$LINE\"" 1; return 1; }
}

# Expand a binary pickle (optionally compressed with zstd or zlib) to python, recognising its format from its first
# bytes.
function dumpBinary() {
    python3 - "$1" <<@EOF
import pickle
import sys
import zlib
if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd
data = open(sys.argv[1], "rb").read()
if data.startswith(b"\x28\xb5\x2f\xfd"):
    data = zstd.decompress(data)
elif data.startswith(b"\x78"):
    data = zlib.decompress(data)
# like "edmConfigDump -o", without a trailing newline
sys.stdout.write(pickle.loads(data).dumpPython())
@EOF
}

# Compare a binary pickle with the reference.
function checkBinary() {
    TEST="$1"
    OUT="$2"
    dumpBinary $OUT > dump_test$TEST.py 2> dump_test$TEST.log || { fail "Test $TEST: failure reading $OUT" $?; return 1; }
    (diff $REFOUT dump_test$TEST.py) || { fail "Test $TEST: incorrect content of $OUT" $?; return 1; }
}

# Run edmConfigPickle, writing a binary pickle, check its first bytes, and compare it with the reference.
function doBinaryTest() {
    TEST="$1"
    CMD="$2"
    OUT="$3"
    MAGIC="$4"
    LOG="log_test$TEST.log"
    $CMD >& $LOG || { fail "Test $TEST: failure running $CMD" $?; return 1; }
    if [ "$(od -A n -t x1 -N ${#MAGIC} $OUT | tr -d ' \n' | cut -c1-${#MAGIC})" != "$MAGIC" ]; then
        fail "Test $TEST: $OUT does not start with $MAGIC" 1
        return 1
    fi
    checkBinary $TEST $OUT
}

# Run edmConfigPickle with invalid options, and check that it fails with an error message, without writing any output.
function doFailTest() {
    TEST="$1"
    CMD="$2"
    LOG="log_test$TEST.log"
    OUT="out_test$TEST"
    if $CMD > $OUT 2> $LOG; then
        fail "Test $TEST: $CMD should have failed" 1
        return 1
    fi
    grep -q "error:" $LOG || { fail "Test $TEST: no error message from $CMD" 1; return 1; }
    if [ -s $OUT ]; then
        fail "Test $TEST: $CMD should not have written any output" 1
        return 1
    fi
}

# test edmConfigPickle w/ argparse, with the default options: zstd level 6, base64 encoded, to standard output;
# the arguments after the configuration file (including -o) are passed to the configuration
function test1() {
    OUT=pickle_argparse_cfg.py
    edmConfigPickle ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2 > $OUT 2> log_test1.log || { fail "Test 1: failure running edmConfigPickle" $?; return 1; }
    checkDump 1 $OUT || return 1
    checkLine 1 $OUT "^# Configuration test_argparse.py -o foo -i 2, pickled and compressed with zstd level 6\.$"
}

# test edmConfigPickle w/ varparsing
function test2() {
    OUT=pickle_varparsing_cfg.py
    doTest 2 "edmConfigPickle -o $OUT ${LOCAL_TEST_DIR}/test_varparsing.py output=foo intprod=2" $OUT
}

# test the compression options
function test3() {
    OUT=pickle_zlib9_cfg.py
    doTest 3 "edmConfigPickle -z -9 -o $OUT ${LOCAL_TEST_DIR}/test_varparsing.py output=foo intprod=2" $OUT || return 1
    checkLine 3 $OUT "pickled and compressed with zlib level 9\.$"
}

function test4() {
    OUT=pickle_zstd19_cfg.py
    doTest 4 "edmConfigPickle -19 --zstd -o $OUT ${LOCAL_TEST_DIR}/test_varparsing.py output=foo intprod=2" $OUT || return 1
    checkLine 4 $OUT "pickled and compressed with zstd level 19\.$"
}

function test5() {
    OUT=pickle_uncompressed_cfg.py
    doTest 5 "edmConfigPickle -0 -z -o $OUT ${LOCAL_TEST_DIR}/test_varparsing.py output=foo intprod=2" $OUT || return 1
    checkLine 5 $OUT "pickled and uncompressed\.$" || return 1
    if grep -q "decompress" $OUT; then
        fail "Test 5: $OUT should not decompress the pickle" 1
        return 1
    fi
}

# the last compression level wins
function test6() {
    OUT=pickle_levels_cfg.py
    doTest 6 "edmConfigPickle -3 -1 --zlib -o $OUT ${LOCAL_TEST_DIR}/test_varparsing.py output=foo intprod=2" $OUT || return 1
    checkLine 6 $OUT "pickled and compressed with zlib level 1\.$"
}

# test the binary pickles: zstd (default), zlib and uncompressed;
# the second byte of a zlib stream encodes the compression level: 01 for level 1, da for level 9
function test7() {
    OUT=pickle_argparse.pkl.zst
    doBinaryTest 7 "edmConfigPickle -p -o $OUT ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2" $OUT 28b52ffd
}

function test8() {
    OUT=pickle_argparse_9.pkl.zlib
    doBinaryTest 8 "edmConfigPickle --pickle --zlib -9 -o $OUT ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2" $OUT 78da
}

function test9() {
    OUT=pickle_argparse_1.pkl.zlib
    doBinaryTest 9 "edmConfigPickle -p -z -1 -o $OUT ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2" $OUT 7801
}

function test10() {
    OUT=pickle_argparse.pkl
    doBinaryTest 10 "edmConfigPickle -p -0 -o $OUT ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2" $OUT 80
}

# a binary pickle is written to standard output only with "-o -"
function test11() {
    doFailTest 11 "edmConfigPickle -p ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2"
}

function test12() {
    OUT=pickle_stdout.pkl.zst
    edmConfigPickle -p -o - ${LOCAL_TEST_DIR}/test_argparse.py -o foo -i 2 > $OUT 2> log_test12.log || { fail "Test 12: failure running edmConfigPickle -p -o -" $?; return 1; }
    checkBinary 12 $OUT
}

# invalid compression levels and incompatible options
function test13() {
    doFailTest 13 "edmConfigPickle -z -10 ${LOCAL_TEST_DIR}/test_argparse.py"
}

function test14() {
    doFailTest 14 "edmConfigPickle -20 ${LOCAL_TEST_DIR}/test_argparse.py"
}

function test15() {
    doFailTest 15 "edmConfigPickle -z -s ${LOCAL_TEST_DIR}/test_argparse.py"
}

function test16() {
    doFailTest 16 "edmConfigPickle -b -p -o pickle_fail.pkl ${LOCAL_TEST_DIR}/test_argparse.py" || return 1
    if [ -e pickle_fail.pkl ]; then
        fail "Test 16: pickle_fail.pkl should not have been written" 1
        return 1
    fi
}

# anything printed by the configuration goes to stderr, not to the output; no bytecode is written next to it
function test17() {
    mkdir -p print_test
    cat > print_test/test_print.py <<@EOF
import FWCore.ParameterSet.Config as cms
print("printed by the configuration")
process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
@EOF
    OUT=pickle_print_cfg.py
    LOG=log_test17.log
    edmConfigPickle print_test/test_print.py > $OUT 2> $LOG || { fail "Test 17: failure running edmConfigPickle" $?; return 1; }
    if grep -q "printed by the configuration" $OUT; then
        fail "Test 17: the output of the configuration ended up in $OUT" 1
        return 1
    fi
    checkLine 17 $LOG "printed by the configuration" || return 1
    python3 -c "exec(open('$OUT').read()); assert process.name_() == 'TEST'" || { fail "Test 17: $OUT is not a valid configuration" 1; return 1; }
    if [ -e print_test/__pycache__ ]; then
        fail "Test 17: edmConfigPickle wrote bytecode next to the configuration" 1
        return 1
    fi
}

TESTS=17
for N in $(seq 1 $TESTS); do
    test$N
done

if [ $FAILED -ne 0 ]; then
    echo "$FAILED out of $TESTS tests failed"
    exit 1
fi
echo "All $TESTS tests passed"
