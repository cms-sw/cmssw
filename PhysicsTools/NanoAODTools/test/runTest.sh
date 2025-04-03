#!/bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "===== Test NanoAODTools functionality ===="

(${SCRAM_TEST_PATH}/exampleAnalysis.py)      || die "Falure in exampleAalysis" $?
(${SCRAM_TEST_PATH}/example_postproc.py)     || die "Falure in example_postproc" $?
