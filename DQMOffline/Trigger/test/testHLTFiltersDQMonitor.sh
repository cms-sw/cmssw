#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  exit $2
}

# run test job
TESTDIR="${LOCALTOP}"/src/DQMOffline/Trigger/test

cmsRun "${TESTDIR}"/testHLTFiltersDQMonitor_cfg.py -- -t 4 -n 128 \
  || die "Failure running testHLTFiltersDQMonitor_cfg.py" $?
