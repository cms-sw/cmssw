#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING BeamSpotOnline From DB Read / Write codes ..."

## clean the input db files
if test -f "test_BSHLT_tag.db"; then
    rm -fr test_BSHLT_tag.db
fi

if test -f "test_BSLegacy_tag.db"; then
    rm -fr test_BSLegacy_tag.db
fi

## copy the input file
cp -pr $CMSSW_BASE/src/CondTools/BeamSpot/data/BeamFitResults_Run306171.txt .

# test write
printf "TESTING Writing BeamSpotOnlineLegacyObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsWriter_cfg.py unitTest=True inputRecord=BeamSpotOnlineLegacyObjectsRcd || die "Failure writing payload for BeamSpotOnlineLegacyObjectsRcd" $?

printf "TESTING Writing BeamSpotOnlineHLTObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsWriter_cfg.py unitTest=True inputRecord=BeamSpotOnlineHLTObjectsRcd || die "Failure writing payload for BeamSpotOnlineHLTObjectsRcd" $?
# test read

printf "TESTING Reading BeamSpotOnlineLegacyObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsReader_cfg.py unitTest=True inputRecord=BeamSpotOnlineLegacyObjectsRcd || die "Failure reading payload for BeamSpotOnlineLegacyObjectsRcd" $?

printf "TESTING Reading BeamSpotOnlineHLTObjectsRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineRecordsReader_cfg.py unitTest=True inputRecord=BeamSpotOnlineHLTObjectsRcd || die "Failure reading payload for BeamSpotOnlineHLTObjectsRcd" $?

echo "TESTING reading BeamSpotObjectRcd DB object ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/BeamSpotRcdPrinter_cfg.py || die "Failure running BeamSpotRcdPrinter" $?
