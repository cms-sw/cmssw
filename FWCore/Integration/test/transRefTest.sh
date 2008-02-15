#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/TransRefTest.cfg || die 'Failed in TransRefTest.cfg' $?

# Pass in name and status



