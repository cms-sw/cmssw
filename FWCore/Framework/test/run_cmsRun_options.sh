#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

CONFIG="import FWCore.ParameterSet.Config as cms
process = cms.Process('Test')
process.source = cms.Source('EmptySource')
process.maxEvents.input = 10"
echo "$CONFIG" > test_cmsRun_options_cfg.py

# expect cmsRun to succeed, and the log to contain all the given patterns
function expectSuccess {
  local TEST="$1"; shift
  local LOG="log_test_options_$TEST.log"
  local -a ARGS=()
  while [ "$1" != "--" ]; do ARGS+=("$1"); shift; done; shift
  cmsRun "${ARGS[@]}" >& $LOG || die "Test $TEST: failure running cmsRun ${ARGS[*]}" 1
  for PATTERN in "$@"; do
    grep -qF "$PATTERN" $LOG || die "Test $TEST: missing \"$PATTERN\" in the output of cmsRun ${ARGS[*]}" 1
  done
}

# expect cmsRun to fail, and the log to contain all the given patterns
function expectFailure {
  local TEST="$1"; shift
  local LOG="log_test_options_$TEST.log"
  local -a ARGS=()
  while [ "$1" != "--" ]; do ARGS+=("$1"); shift; done; shift
  cmsRun "${ARGS[@]}" >& $LOG && die "Test $TEST: unexpected success running cmsRun ${ARGS[*]}" 1
  for PATTERN in "$@"; do
    grep -qF "$PATTERN" $LOG || die "Test $TEST: missing \"$PATTERN\" in the output of cmsRun ${ARGS[*]}" 1
  done
}

# options of different types, from a configuration file
expectSuccess file \
  -o "dumpOptions = True" -o "numberOfThreads=4" -o "numberOfStreams = 3" -o "sizeOfStackForThreadsInKB = 10240" \
  -o 'accelerators = ["cpu"]' -o 'Rethrow = "ProductNotFound", "InvalidReference"' -o 'fileMode = "NOMERGE"' -o "eventSetup.numberOfConcurrentIOVs = 1" \
  test_cmsRun_options_cfg.py -- \
  "Number of Threads = 4" "Number of Streams = 3" "Number of Concurrent IOVs = 1"

# options from a configuration passed on the command line
expectSuccess command -c "$CONFIG" -o "dumpOptions=True" -o "numberOfThreads=2" -- \
  "Number of Threads = 2" "Number of Streams = 2"

# the last value of a repeated option is used
expectSuccess repeated -o "dumpOptions=True" -o "numberOfThreads=2" -o "numberOfThreads=3" test_cmsRun_options_cfg.py -- \
  "Number of Threads = 3"

# -n takes precedence over -o numberOfThreads
expectSuccess numThreads -n 4 -o "dumpOptions=True" -o "numberOfThreads=2" test_cmsRun_options_cfg.py -- \
  "Number of Threads = 4"

# a single value can be given for a list option; if it were split into characters, 'c' would not be a valid accelerator
expectSuccess single -o 'accelerators="cpu"' -o 'Rethrow="ProductNotFound"' -o "dumpOptions=True" test_cmsRun_options_cfg.py -- \
  "Number of Threads = 1"

# missing '=' between name and value
expectFailure syntax -o "numberOfThreads" test_cmsRun_options_cfg.py -- \
  "invalid process option 'numberOfThreads', the syntax is '-o name=value'"

# unknown option
expectFailure unknown -o "unknown = 42" test_cmsRun_options_cfg.py -- \
  '"unknown" is not a valid process option'

# value of the wrong type
expectFailure type -o 'numberOfThreads = "four"' test_cmsRun_options_cfg.py -- \
  'Invalid process option "numberOfThreads = "four""' "ValueError"

# value that is not a valid python expression
expectFailure value -o "fileMode = NOMERGE" test_cmsRun_options_cfg.py -- \
  'Invalid process option "fileMode = NOMERGE"' "NameError"

# single value of the wrong type for a list option
expectFailure single_type -o "accelerators = 42" test_cmsRun_options_cfg.py -- \
  'Invalid process option "accelerators = 42"' "TypeError"

# unknown nested option
expectFailure nested -o "eventSetup.unknown = 42" test_cmsRun_options_cfg.py -- \
  'Invalid process option "eventSetup.unknown = 42"'

exit 0
