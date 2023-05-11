#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  exit $2
}

# run test job
cmsRun "${SCRAM_TEST_PATH}"/testTriggerMonitors_dqm_cfg.py \
  || die "Failure running testTriggerMonitors_dqm_cfg.py" $?

cmsRun "${SCRAM_TEST_PATH}"/testTriggerMonitors_harvesting_cfg.py \
  || die "Failure running testTriggerMonitors_harvesting_cfg.py" $?
