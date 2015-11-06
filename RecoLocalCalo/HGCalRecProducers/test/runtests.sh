#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/testHGCalRecoLocal_cfg.py || die 'Failure using testHGCalRecoLocal_cfg.py' $?