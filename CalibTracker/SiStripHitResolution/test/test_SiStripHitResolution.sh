#!/bin/bash
function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

echo -e "Testing SiStripHitEfficencyWorker \n\n"
cmsRun ${SCRAM_TEST_PATH}/SiStripHitResol_test.py isUnitTest=True || die 'failed running SiStripHitResol_test.py' $?

echo -e "Testing CPEanconfig.py \n\n"
cmsRun ${SCRAM_TEST_PATH}/CPEanconfig.py isUnitTest=True || die 'failed running CPEanconfig.py' $?

echo -e "Testing SiStripHitResolutionFromCalibTree_cfg.py \n\n"
cmsRun ${SCRAM_TEST_PATH}/SiStripHitResolutionFromCalibTree_cfg.py || die 'failed running SiStripHitResolutionFromCalibTree_cfg.py' $?

### To be implemented

#echo -e "Testing SiStripHitEfficencyHarvester \n\n"
#cmsRun ${SCRAM_TEST_PATH}/testHitEffHarvester.py isUnitTest=True || die 'failed running testHitEffHarvester.py' $?

#echo -e " testing tSiStripHitEffFromCalibTree \n\n"
#cmsRun ${SCRAM_TEST_PATH}/testSiStripHitEffFromCalibTree_cfg.py inputFiles=HitEffTree.root runNumber=325172 || die 'failed running testSiStripHitEffFromCalibTree_cfg.py' $?

echo -e "Done with the tests!"
