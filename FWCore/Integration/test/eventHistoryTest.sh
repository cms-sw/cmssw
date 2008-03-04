#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

# Write a file for the FIRST process
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventHistory_1.cfg || die 'Failed in EventHistory_1' $?
echo "*************************************************"
echo "**************** Finished pass 1 ****************"
echo "*************************************************"

# Read the first file, make the SECOND process
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventHistory_2.cfg || die 'Failed in EventHistory_1' $?
echo "*************************************************"
echo "**************** Finished pass 2 ****************"
echo "*************************************************"

# Read the second file, make sure the event data have the right history
cmsRun --parameter-set ${LOCAL_TEST_DIR}/EventHistory_3.cfg || die 'Failed in EventHistory_1' $?
echo "*************************************************"
echo "**************** Finished pass 3 ****************"
echo "*************************************************"





