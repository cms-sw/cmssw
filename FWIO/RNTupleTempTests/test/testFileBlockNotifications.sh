#!/bin/bash

function die { echo "Failure $1: status $2"; exit "$2"; }

function run {
  local STAGE="$1"
  shift
  cmsRun "${SCRAM_TEST_PATH}/testFileBlockNotification_cfg.py" --stage "$STAGE" "$@" || die "$STAGE" $?
}

run write --output fileBlockNotification_first.root
run merge --input fileBlockNotification_first.root --output fileBlockNotification_merged.root
run read --input fileBlockNotification_merged.root
