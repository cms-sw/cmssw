#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

if test -f "SiStripConditionsDBFile.db"; then
    echo "cleaning the local test area"
    rm -fr SiStripConditionsDBFile.db  # builders test
    rm -fr modifiedSiStrip*.db         # miscalibrator tests
fi

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

## do the miscalibrators
(cmsRun ${LOCAL_TEST_DIR}/SiStripChannelGainFromDBMiscalibrator_cfg.py) || die "Failure using cmsRun SiStripChannelGainFromDBMiscalibrator_cfg.py)" $?
(cmsRun ${LOCAL_TEST_DIR}/SiStripNoiseFromDBMiscalibrator_cfg.py) || die "Failure using cmsRun SiStripNoiseFromDBMiscalibrator_cfg.py" $?

echo -e " Done with the miscalibrators \n\n"
