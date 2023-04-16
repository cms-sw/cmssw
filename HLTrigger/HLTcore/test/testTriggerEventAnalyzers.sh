#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  exit $2
}

# run test job
cmsRun "${SCRAM_TEST_PATH}"/testTriggerEventAnalyzers_cfg.py \
  || die "Failure running testTriggerEventAnalyzers_cfg.py" $?
