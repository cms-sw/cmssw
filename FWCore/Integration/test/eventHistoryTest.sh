#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

# Write a file for the FIRST process
cmsRun ${LOCAL_TEST_DIR}/EventHistory_1_cfg.py || die 'Failed in EventHistory_1' $?
echo "*************************************************"
echo "**************** Finished pass 1 ****************"
echo "*************************************************"

# Read the first file, write the second.
cmsRun ${LOCAL_TEST_DIR}/EventHistory_2_cfg.py || die 'Failed in EventHistory_2' $?
echo "*************************************************"
echo "**************** Finished pass 2 ****************"
echo "*************************************************"

# Read the second file, write the third.
cmsRun ${LOCAL_TEST_DIR}/EventHistory_3_cfg.py || die 'Failed in EventHistory_3' $?
echo "*************************************************"
echo "**************** Finished pass 3 ****************"
echo "*************************************************"

# Read the third file, make sure the event data have the right history
cmsRun ${LOCAL_TEST_DIR}/EventHistory_4_cfg.py || die 'Failed in EventHistory_4' $?
echo "*************************************************"
echo "**************** Finished pass 4 ****************"
echo "*************************************************"

# Read the fourth file, make sure the event data have the right history
cmsRun ${LOCAL_TEST_DIR}/EventHistory_5_cfg.py || die 'Failed in EventHistory_5' $?
echo "*************************************************"
echo "**************** Finished pass 5 ****************"
echo "*************************************************"

# Read the fifth file, make sure the event data have the right history
cmsRun ${LOCAL_TEST_DIR}/EventHistory_6_cfg.py || die 'Failed in EventHistory_6' $?
echo "*************************************************"
echo "**************** Finished pass 6 ****************"
echo "*************************************************"

# Repeat all the above steps, but slightly modified so all steps run in one job with subprocess
cmsRun ${LOCAL_TEST_DIR}/EventHistory_SubProcess_cfg.py || die 'Failed in EventHistory_SubProcess' $?
echo "*************************************************"
echo "**************** Finished pass SubProcess *******"
echo "*************************************************"
