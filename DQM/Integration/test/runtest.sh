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
    CLIENTS_DIR=$LOCAL_TEST_DIR/src/DQM/Integration/python/clients
fi

mkdir -p $LOCAL_TEST_DIR/upload
cmsRun $CLIENTS_DIR/$1 unitTest=True
