#!/bin/sh
function die { echo $1: status $2 ; exit $2; }

# test worker
printf "TESTING SiStrip Lorentz Angle Worker ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/step_PromptCalibProdSiStripLA_cfg.py || die "Failure running step_PromptCalibProdSiStripLA_cfg.py" $?

# test harvester
printf "TESTING SiStrip Lorentz Angle Harvester ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/step_PromptCalibProdSiStripLA_ALCAHARVEST_cfg.py || die "Failure running step_PromptCalibProdSiStripLA_ALCAHARVEST_cfg.py" $?
