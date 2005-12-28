#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/inputSourceTest.cfg || die 'Failed in inputSourceTest.cfg' $?

# Pass in name and status



