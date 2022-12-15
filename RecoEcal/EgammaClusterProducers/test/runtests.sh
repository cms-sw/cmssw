#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

if ! singularity-check.sh; then
  echo "missing singularity or missing unprivileged user namespace support"
  exit 0
fi

cmsRun ${LOCAL_TEST_DIR}/DRNTest_cfg.py || die 'Failure using cmsRun' $?
