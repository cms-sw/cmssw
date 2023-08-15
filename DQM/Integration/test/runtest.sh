#!/bin/bash
set -e
set -x

if [[ $# -eq 0 ]]; then
    echo "Please provide a name of the client"
    exit 1
fi

if [[ -z ${LOCAL_TEST_DIR} ]]; then
    LOCAL_TEST_DIR=.
fi

if [[ -z ${CLIENTS_DIR} ]]; then
    CLIENTS_DIR=${CMSSW_BASE}/src/DQM/Integration/python/clients
fi

mkdir -p $LOCAL_TEST_DIR/upload

if [[ $# -eq 1 ]]; then
    cmsRun $CLIENTS_DIR/$1 unitTest=True
else
    echo "Will use streamers files for run $2"
    cmsRun $CLIENTS_DIR/$1 unitTest=True runNumber=$2
fi
