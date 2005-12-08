#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupTest.cfg || die 'Failed in EventSetupTest.cfg' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventSetupTest2.cfg || die 'Failed in EventSetupTest2.cfg' $?
