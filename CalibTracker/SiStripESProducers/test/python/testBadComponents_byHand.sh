#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING CalibTracker/SiStripESProducers ..."
cmsRun ${SCRAM_TEST_PATH}/python/SiStripBadAPVListBuilder_byHand_cfg.py || die "Failure running SiStripBadAPVListBuilder_byHand_cfg.py" $?
