#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/parentlessTest.cfg || die 'Failed in parentlessTest.cfg' $?

# Pass in name and status



