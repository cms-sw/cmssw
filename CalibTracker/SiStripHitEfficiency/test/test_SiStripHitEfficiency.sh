#!/bin/bash
function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

echo -e "Testing SiStripHitEfficencyWorker \n\n"
cmsRun ${LOCAL_TEST_DIR}/testHitEffWorker.py isUnitTest=True || die 'failed running testHitEffWorker.py' $?

echo -e "Testing SiStripHitEfficencyHarvester \n\n"
cmsRun ${LOCAL_TEST_DIR}/testHitEffHarvester.py isUnitTest=True || die 'failed running testHitEffHarvester.py' $?

echo -e " testing tSiStripHitEffFromCalibTree \n\n"
cmsRun ${LOCAL_TEST_DIR}/testSiStripHitEffFromCalibTree_cfg.py inputFiles=HitEffTree.root runNumber=325172 || die 'failed running testSiStripHitEffFromCalibTree_cfg.py' $?

echo -e "Done with the tests!"
