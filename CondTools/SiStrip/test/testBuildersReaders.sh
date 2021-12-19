#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi
if test -f "SiStripConditionsDBFile.db"; then
    echo "cleaning the local test area"
    rm -fr SiStripConditionsDBFile.db  # builders test
    rm -fr modifiedSiStrip*.db         # miscalibrator tests
fi
pwd
echo " testing CondTools/SiStrip"

## do the builders first (need the input db file)
for entry in "${LOCAL_TEST_DIR}/"SiStrip*Builder_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the writers \n\n"

## do the readers
for entry in "${LOCAL_TEST_DIR}/"SiStrip*Reader_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the readers \n\n"

sleep 5

## do the miscalibrators
for entry in "${LOCAL_TEST_DIR}/"SiStrip*Miscalibrator_cfg.py
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

echo -e " Done with the miscalibrators \n\n"

## do the scaler (commented for now)
#(cmsRun ${LOCAL_TEST_DIR}/SiStripApvGainRescaler_cfg.py) || die "Failure using cmsRun SiStripApvGainRescaler_cfg.py)" $?
#echo -e " Done with the gain rescaler \n\n"
