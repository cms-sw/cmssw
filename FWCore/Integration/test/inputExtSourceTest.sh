#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/inputExtSourceTest.cfg || die 'Failed in inputExtSourceTest.cfg' $?

# Pass in name and status



