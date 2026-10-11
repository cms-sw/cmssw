#!/bin/bash
# Test the pipelined decoding of NewEventStreamFileReader (numDecodeTasks > 1 or 0) against the
# serial one (numDecodeTasks = 1): several input files, a change of the event meta data in the
# middle of a file, repeated meta data, skipping events, and errors reading or decoding the last
# event. Note: skipEvents is applied before any event is read ahead, so the skipping of events
# already read ahead is not exercised here.

function die { echo Failure $1: status $2 ; echo ""; cat log ; exit $2 ; }
# number of events processed by a successful job, and number of events started by a failed one
function processed { grep -m1 "TrigReport Events total" $1 | awk '{print $5}'; }
function started { grep -c "Begin processing the" $1; }

N=50
TEST=${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamerTest_cfg.py

# two files with different ProductID meta data, and their concatenation with a single INIT message
cmsRun ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamer_cfg.py --maxEvents $N > log 2>&1 || die "producer 1" $?
cmsRun ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamer_cfg.py --maxEvents $N --enableOther > log 2>&1 || die "producer 2" $?
CatStreamerFiles refconsistency_cat.dat refconsistency_1.dat refconsistency_10.dat > log 2>&1 || die "CatStreamerFiles" $?
# a file with a repeated meta data message with the same checksum, as in merged HLT files
CatStreamerFiles refconsistency_same.dat refconsistency_1.dat refconsistency_1.dat > log 2>&1 || die "CatStreamerFiles" $?

# copies of the concatenated file with the last event truncated, and with a wrong checksum
python3 - <<'PYEOF'
data = open('refconsistency_cat.dat', 'rb').read()
open('refconsistency_truncated.dat', 'wb').write(data[:-10])
corrupted = bytearray(data)
corrupted[-10] ^= 0xff
open('refconsistency_corrupted.dat', 'wb').write(corrupted)
PYEOF

function expect_success {   # expected events, options...
  local expected=$1; shift
  cmsRun $TEST "$@" > log 2>&1 || die "cmsRun $TEST $*" $?
  local n=$(processed log)
  [ "$n" = "$expected" ] || die "cmsRun $TEST $*: processed $n events instead of $expected" 1
  echo "ok: $* -> $n events"
}

function expect_failure {   # expected started events (checked with one thread), error message, options...
  local expected=$1 message=$2; shift 2
  cmsRun $TEST "$@" > log 2>&1 && die "cmsRun $TEST $* (expected to fail)" 1
  grep -q "$message" log || die "cmsRun $TEST $*: failed without '$message'" 1
  local n=$(started log)
  # with several streams, events already read may not have started when the job stops
  if [[ " $* " == *" --threads 1 "* ]]; then
    [ "$n" = "$expected" ] || die "cmsRun $TEST $*: started $n events instead of $expected before the error" 1
  fi
  echo "ok: $* -> failed after $n events"
}

for opts in "--numDecodeTasks 1 --threads 1" "--numDecodeTasks 1 --threads 4" \
            "--numDecodeTasks 16 --threads 1" "--numDecodeTasks 0 --threads 4" \
            "--numDecodeTasks 16 --threads 4"; do
  expect_success $((2*N)) $opts --input refconsistency_1.dat --input refconsistency_10.dat
  expect_success $((2*N)) $opts --input refconsistency_cat.dat
  expect_success $((2*N)) $opts --input refconsistency_same.dat
  expect_success $((2*N-7)) $opts --input refconsistency_cat.dat --skipEvents 7
  # skipEvents stops at the end of a file, the same as in serial mode
  expect_success $N $opts --input refconsistency_1.dat --input refconsistency_10.dat --skipEvents $((N+5))
  # the events before a read or decoding error are all processed
  expect_failure $((2*N-1)) "FileReadError\|Failed reading streamer file" $opts --input refconsistency_truncated.dat
  expect_failure $((2*N-1)) "Checksum error" $opts --input refconsistency_corrupted.dat
done

exit 0
