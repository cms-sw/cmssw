#!/bin/bash

 function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

echo -e " Tesing SiStripHashedDetId \n\n"

cmsRun ${SCRAM_TEST_PATH}/testSiStripHashedDetId_cfg.py || die "Failure running testSiStripHashedDetId_cfg.py" $?
