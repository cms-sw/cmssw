#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/RefTest.cfg || die 'Failed in RefTest.cfg' $?

# Pass in name and status



