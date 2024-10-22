#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

THIS_ARCH=$(echo $SCRAM_ARCH | cut -d'_' -f2)
if [ "$THIS_ARCH" == "amd64" ]; then
	echo "has amd64"
else
	echo "missing amd64"
	exit 0
fi

if ! apptainer-check.sh; then
	echo "missing apptainer/singularity or missing unprivileged user namespace support"
	exit 0
fi

cmsRun ${SCRAM_TEST_PATH}/DRNTest_cfg.py || die 'Failure using cmsRun' $?
