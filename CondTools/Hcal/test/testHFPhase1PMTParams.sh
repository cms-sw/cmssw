#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING HFPhase1PMTParams generation code ..."
write_HFPhase1PMTParams 1 HFPhase1PMTParams_V00_mc.bbin || die "Failure running write_HFPhase1PMTParams" $?

echo "TESTING HFPhase1PMTParams database creation code ..."
cmsRun ${SCRAM_TEST_PATH}/HFPhase1PMTParamsDBWriter_cfg.py || die "Failure running HFPhase1PMTParamsDBWriter_cfg.py" $?

echo "TESTING HFPhase1PMTParams database reading code ..."
cmsRun ${SCRAM_TEST_PATH}/HFPhase1PMTParamsDBReader_cfg.py || die "Failure running HFPhase1PMTParamsDBReader_cfg.py" $?

echo "TESTING that we can restore HFPhase1PMTParams from the database ..."
diff HFPhase1PMTParams_V00_mc.bbin dbread.bbin || die "Database contents differ from the original" $?
