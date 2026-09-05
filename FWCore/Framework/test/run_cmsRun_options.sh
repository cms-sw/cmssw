#!/bin/bash -x

LOCAL_TEST_DIR="$CMSSW_BASE/src/FWCore/Framework/test"
source "$LOCAL_TEST_DIR/help_cmsRun_tests.sh"

# do these manually because quote nesting becomes a nightmare / perhaps actually impossible

# Test config, passed as command line input
CONFIG_INPUT="import FWCore.ParameterSet.Config as cms
process = cms.Process('Test')
process.source = cms.Source('EmptySource')
process.maxEvents.input = 10"

# Check that options of different types can be set
TEST=parse_options
LOG="log_test_$TEST.log"
cmsRun \
  -c "$CONFIG_INPUT" \
  -o "dumpOptions = True" \
  -o "numberOfStreams = 4" \
  -o "numberOfThreads = 4" \
  -o "sizeOfStackForThreadsInKB = 10240" \
  -o 'accelerators = "cpu", "gpu-*"' \
  -o 'fileMode = "NOMERGE"' \
  >& $LOG || die "Test $TEST: unexpected failure running cmsRun -c \"$CONFIG_INPUT\""

# Check that the parser rejects options that do not follow the 'name = value' syntax
TEST=invalid_option
LOG="log_test_$TEST.log"
cmsRun \
  -c "$CONFIG_INPUT" \
  -o "invalid" \
  >& $LOG && die "Test $TEST: unexpected success running cmsRun -c \"$CONFIG_INPUT\" -o \"invalid\""
(grep -qF "missing '=' separator between name and value" $LOG) || die "Test $TEST: incorrect output from cmsRun -c \"$CONFIG_INPUT\" -o \"invalid\""

# Check that the parser rejects non-existing options
TEST=unknown_options
LOG="log_test_$TEST.log"
cmsRun \
  -c "$CONFIG_INPUT" \
  -o "unknown = 42" \
  >& $LOG && die "Test $TEST: unexpected success running cmsRun -c \"$CONFIG_INPUT\" -o \"unknown = 42\""
(grep -qF "is not a valid process option" $LOG) || die "Test $TEST: incorrect output from cmsRun -c \"$CONFIG_INPUT\" -o \"unknown = 42\""

# The following tests use non-existing options to test arbitrary types.
# The jobs are expected to fail with an "Illegal parameter found in configuration" message,
# but that happens after the command line options have been processed and validated.

# Check bool option values
TYPE=bool
LOG="log_test_$TYPE.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE(False)"
OPTION="unused = False"
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

# Check int32 option values
TYPE=int32
LOG="log_test_$TYPE.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE(0)"
OPTION="unused = -42"
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

# Check uint32 option values
TYPE=uint32
LOG="log_test_$TYPE.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE(0)"
OPTION="unused = +42"
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

# Check int64 option values
TYPE=int64
LOG="log_test_$TYPE.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE(0)"
OPTION="unused = 42 "
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

# Check uint64 option values
TYPE=uint64
LOG="log_test_$TYPE.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE(0)"
OPTION="unused = -42"
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

# Check double option values
TYPE=double
LOG="log_test_$TYPE.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE(0.)"
OPTION="unused = 42.e+0"
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

# Check string option values
TYPE=string
LOG="log_test_${TYPE}_single.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE('')"
OPTION="unused = 'single quoted'"
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

# Check string option values
TYPE=string
LOG="log_test_${TYPE}_double.log"
UNUSED_CONFIG_INPUT="$CONFIG_INPUT
process.options.unused = cms.untracked.$TYPE('')"
OPTION='unused = "double quoted"'
cmsRun \
  -c "$UNUSED_CONFIG_INPUT" \
  -o "$OPTION" \
  >& $LOG && die "Test $TYPE: unexpected success running cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""
(grep -qF "Illegal parameter found in configuration" $LOG) || die "Test $TYPE: incorrect output from cmsRun -c \"$UNUSED_CONFIG_INPUT\" -o \"$OPTION\""

