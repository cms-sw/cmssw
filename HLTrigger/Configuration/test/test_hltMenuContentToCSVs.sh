#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  if [ $# -gt 2 ]; then
    printf "%s\n" "=== Log File =========="
    cat $3
    printf "%s\n" "=== End of Log File ==="
  fi
  exit $2
}

if [ -z "${SCRAM_TEST_PATH}" ]; then
  printf "\n%s\n\n" "ERROR -- environment variable SCRAM_TEST_PATH not defined"
  exit 1
fi

# run test job
exeDir="${SCRAM_TEST_PATH}"/../scripts/utils

"${exeDir}"/hltMenuContentToCSVs /dev/CMSSW_13_0_0/GRun \
  --meta "${exeDir}"/hltPathOwners.json &> test_hltMenuContentToCSVs_log \
  || die "Failure running hltMenuContentToCSVs" $? test_hltMenuContentToCSVs_log
