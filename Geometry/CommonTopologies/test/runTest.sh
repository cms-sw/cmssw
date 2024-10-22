#!/bin/sh
echo " testing Geometry/CommonTopologies"

export tmpdir=${PWD}
cmsRun ${SCRAM_TEST_PATH}/python/ValidateRadial_cfg.py | grep "ALIVE" > /dev/null
if [[ $? -ne 0 ]]
then
 echo "Vin test tricks do not work"
 exit 20
fi
cmsRun ${SCRAM_TEST_PATH}/python/ValidateRadial_cfg.py | grep "BOGUS" > /dev/null
if [[ $? -eq 0 ]]
then
 echo "Vin test tricks do not work"
 exit 20
fi

rm -f failureLimits.root
cmsRun ${SCRAM_TEST_PATH}/python/ValidateRadial_cfg.py | grep "FAILED" > /dev/null
if [[ $? -eq 0 ]]
then
 echo "ValidateRadial failed"
 rm -f failureLimits.root
 exit 20
fi
rm -f failureLimits.root
