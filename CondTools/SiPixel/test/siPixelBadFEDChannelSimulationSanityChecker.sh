#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo -e "TESTING SiPixelBadFEDChannelSimulationSanityChecker ..."
cmsRun  ${SCRAM_TEST_PATH}/SiPixelBadFEDChannelSimulationSanityChecker_cfg.py || die "Failure running SiPixelBadFEDChannelSimulationSanityChecker_cfg.py" $?
