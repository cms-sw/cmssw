#!/bin/bash
function die { echo $1: status $2; exit $2; }

customise_commands="process.GlobalTag.toGet = cms.VPSet()\
\nprocess.GlobalTag.toGet.append(cms.PSet(record = cms.string(\"AlCaRecoTriggerBitsRcd\"),tag =  cms.string(\"AlCaRecoHLTpaths_PPS2022_express_v1\"), connect = cms.string(\"frontier://FrontierProd/CMS_CONDITIONS\")))\
\nprocess.GlobalTag.toGet.append(cms.PSet(record = cms.string(\"PPSTimingCalibrationLUTRcd\"),tag =  cms.string(\"PPSDiamondTimingCalibrationLUT_test\"), connect = cms.string(\"frontier://FrontierProd/CMS_CONDITIONS\")))\
\nprocess.ALCARECOPPSCalMaxTracksFilter.TriggerResultsTag = cms.InputTag(\"TriggerResults\",\"\",\"HLTX\")"

INPUTFILE="/store/group/alca_global/pps_alcareco_producer_tests/outputALCAPPS_single.root"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate $INPUTFILE`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    echo "Using file ${INPUTFILE}. Running in ${SCRAM_TEST_PATH}."
    (cmsDriver.py testExpressPPSAlCaRecoProducer -s ALCAPRODUCER:PPSCalMaxTracks,ENDJOB \
    --process ALCARECO \
    --scenario pp \
    --era ctpps_2018 \
    --conditions auto:run3_data_express \
    --data  \
    --datatier ALCARECO \
    --eventcontent ALCARECO \
    --nThreads 8 \
    --number 100 --filein ${INPUTFILE} \
    --fileout file:outputALCAPPS_RECO_express.root \
    --customise_commands="$customise_commands") || die 'failed running test_express_AlCaRecoProducer' $?
else
    die "SKIPPING test, file ${INPUTFILE} not found" 0
fi
