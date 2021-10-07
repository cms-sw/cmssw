#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING BeamSpotOnline From DB Read / Write codes ..."

## clean the input db file
if test -f "test.db"; then
    rm -fr test.db
fi

## copy the input file
cp -pr $CMSSW_BASE/src/CondTools/BeamSpot/data/BeamFitResults_Run306171.txt .

# test write
echo "TESTING Writing BeamSpotOnlineLegacyObjectsRcd DB object ...\n\n"
cmsRun ${LOCAL_TEST_DIR}/BeamSpotOnlineRecordsWriter_cfg.py unitTest=True inputRecord=BeamSpotOnlineLegacyObjectsRcd || die "Failure writing payload for BeamSpotOnlineLegacyObjectsRcd" $? 

echo "TESTING Writing BeamSpotOnlineHLTObjectsRcd DB object ...\n\n"
cmsRun ${LOCAL_TEST_DIR}/BeamSpotOnlineRecordsWriter_cfg.py unitTest=True inputRecord=BeamSpotOnlineHLTObjectsRcd || die "Failure writing payload for BeamSpotOnlineHLTObjectsRcd" $? 
# test read

echo "TESTING Reading BeamSpotOnlineLegacyObjectsRcd DB object ...\n\n"
cmsRun ${LOCAL_TEST_DIR}/BeamSpotOnlineRecordsReader_cfg.py unitTest=True inputRecord=BeamSpotOnlineLegacyObjectsRcd || die "Failure reading payload for BeamSpotOnlineLegacyObjectsRcd" $? 

echo "TESTING Reading BeamSpotOnlineHLTObjectsRcd DB object ...\n\n"
cmsRun ${LOCAL_TEST_DIR}/BeamSpotOnlineRecordsReader_cfg.py unitTest=True inputRecord=BeamSpotOnlineHLTObjectsRcd || die "Failure reading payload for BeamSpotOnlineHLTObjectsRcd" $? 

echo "TESTING reading BeamSpotObjectRcd DB object ...\n\n"
cmsRun ${LOCAL_TEST_DIR}/BeamSpotRcdPrinter_cfg.py || die "Failure running BeamSpotRcdPrinter" $?
