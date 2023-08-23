#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING BeamSpot + BeamSpotOnline + SimBeamSpot From DB Read / Write codes ..."

## clean the input db files
if test -f "test_BSHLT_tag.db"; then
    rm -fr test_BSHLT_tag.db
fi

if test -f "test_BSLegacy_tag.db"; then
    rm -fr test_BSLegacy_tag.db
fi

if test -f "test_simBS_tag.db"; then
    rm -fr test_simBS_tag.db
fi

## copy the input file
cp -pr $CMSSW_BASE/src/CondTools/BeamSpot/data/BeamFitResults_Run306171.txt .

# test write
printf "TESTING Writing BeamSpotOnlineLegacyObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsWriter_cfg.py unitTest=True inputRecord=BeamSpotOnlineLegacyObjectsRcd || die "Failure writing payload for BeamSpotOnlineLegacyObjectsRcd" $?

printf "TESTING Writing BeamSpotOnlineHLTObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsWriter_cfg.py unitTest=True inputRecord=BeamSpotOnlineHLTObjectsRcd || die "Failure writing payload for BeamSpotOnlineHLTObjectsRcd" $?

printf "TESTING Writing SimBeamSpotObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamProfile2DBWriter_cfg.py unitTest=True || die "Failure writing payload for SimBeamSpotObjectsRcd" $?

# test read
printf "TESTING Reading BeamSpotOnlineLegacyObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsReader_cfg.py unitTest=True inputRecord=BeamSpotOnlineLegacyObjectsRcd || die "Failure reading payload for BeamSpotOnlineLegacyObjectsRcd" $?

printf "TESTING Reading BeamSpotOnlineHLTObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsReader_cfg.py unitTest=True inputRecord=BeamSpotOnlineHLTObjectsRcd || die "Failure reading payload for BeamSpotOnlineHLTObjectsRcd" $?

printf "TESTING reading BeamSpotObjectRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotRcdPrinter_cfg.py || die "Failure running BeamSpotRcdPrinter" $?

printf "TESTING converting BeamSpotOnlineObjects from BeamSpotObjects ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineFromOfflineConverter_cfg.py unitTest=True || die "Failure running BeamSpotRcdPrinter" $?

printf "TESTING Reading SimBeamSpotObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamProfile2DBReader_cfg.py unitTest=True || die "Failure reading payload for SimBeamSpotObjectsRcd" $?
