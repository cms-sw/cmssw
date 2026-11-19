#!/bin/bash

function die { echo $1: status $2 ; exit $2; }

cmsRun ${SCRAM_TEST_PATH}/writeCaloClusterSoA_cfg.py || die "Failure using writeCaloClusterSoA_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/readCaloClusterSoA_cfg.py || die "Failure using readCaloClusterSoA_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/readCaloClusterSoA_cfg.py --rntuple || die "Failure using readCaloClusterSoA_cfg.py --rntuple" $?
