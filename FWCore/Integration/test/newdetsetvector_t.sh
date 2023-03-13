#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${SCRAM_TEST_PATH}/newdetsetvector_t_cfg.py || die 'Failed in detsetvector_t_cfg.py' $?
edmEventSize -A -v testDSTV.root

