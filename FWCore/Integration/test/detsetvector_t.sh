#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/detsetvector_t_cfg.py || die 'Failed in detsetvector_t_cfg.py' $?

