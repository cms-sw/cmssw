#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/detsetvector_t_cfg.py || die 'Failed in detsetvector_t_cfg.py' $?

