#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo -e "TESTING SiPixelGainCalibration codes ..."

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

for entry in "${SCRAM_TEST_PATH}/"SiPixelCondObj*Reader_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the readers \n\n"

echo -e "TESTING Reject Noisy and Dead ...\n\n"
cmsRun  ${SCRAM_TEST_PATH}/SiPixelGainCalibrationRejectNoisyAndDead_cfg.py || die "Failure running SiPixelGainCalibrationRejectNoisyAndDead_cfg.py" $?

echo -e " Done with the test \n\n"
