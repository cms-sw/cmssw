#!/bin/sh


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

${LOCAL_TEST_DIR}/RefTest.sh || die 'Failed to create file' $?
root -b -n -q ${LOCAL_TEST_DIR}/event_looping_cint.C || die 'Failed in event_looping_cint.C' $?
root -b -n -q ${LOCAL_TEST_DIR}/chainevent_looping_cint.C || die 'Failed in chainevent_looping_cint.C' $?
#root -b -n -q ${LOCAL_TEST_DIR}/autoload_with_std.C || die 'Failed in autoload_with_std.C' $?
#root -b -n -q ${LOCAL_TEST_DIR}/autoload_with_missing_std.C || die 'Failed in autoload_with_missing_std.C' $?


