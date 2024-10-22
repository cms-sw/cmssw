#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

echo -e "===== testing SiStripApV Gain manipulations =====\n\n"

entries=("SiStripGainPayloadCopyAndExclude_cfg.py" "SiStripApvGainInspector_cfg.py")

echo -e "===== copying IOV 317478 from tag SiStripApvGainAfterAbortGap_PCL_multirun_v0_prompt on dev DB ====="

conddb --yes --db dev copy SiStripApvGainAfterAbortGap_PCL_multirun_v0_prompt SiStripApvGainAAG_pcl --from 317478 --to 317478  --destdb promptCalibConditions86791.db

for entry in "${entries[@]}"; 
do
    echo -e "===== executing cmsRun "${SCRAM_TEST_PATH}/$entry" ======\n"
    (cmsRun "${SCRAM_TEST_PATH}/"$entry) || die "Failure using cmsRun $entry" $?
    echo -e "===== executed $entry test ======\n"
done

echo -e "\n\n ===== Done with the Gain manipulations tests! =====\n\n"
