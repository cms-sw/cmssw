#!/bin/bash

# Pass in name and status
function die {
  echo $1: status $2
  echo === Log file ===
  cat ${3:-/dev/null}
  echo === End log file ===
  exit $2
}

# run test job
TESTDIR="${LOCALTOP}"/src/HLTrigger/HLTcore/test

cmsRun "${TESTDIR}"/testTriggerEventAnalyzers_cfg.py &> log_testTriggerEventAnalyzers \
  || die "Failure running testTriggerEventAnalyzers_cfg.py" $? log_testTriggerEventAnalyzers
