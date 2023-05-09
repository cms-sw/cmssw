#!/bin/sh


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

root -b -n -q ${SCRAM_TEST_PATH}/autoload_with_namespace.C || die 'Failed in autoload_with_namespace.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/autoload_with_std.C || die 'Failed in autoload_with_std.C' $?
root -b -n -q ${SCRAM_TEST_PATH}/autoload_with_missing_std.C || die 'Failed in autoload_with_missing_std.C' $?


