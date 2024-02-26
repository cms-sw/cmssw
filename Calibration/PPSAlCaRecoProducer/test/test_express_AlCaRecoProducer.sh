#!/bin/bash
function die { echo $1: status $2; exit $2; }

# customisation command needed for all tests cases
COMMON_CUSTOM="process.ALCARECOPPSCalMaxTracksFilter.TriggerResultsTag = cms.InputTag(\"TriggerResults\",\"\",\"HLTX\")"

# test on 2022 data
INPUTFILE_355207="/store/data/Run2022B/AlCaPPS/RAW/v1/000/355/207/00000/c23440f4-49c0-44aa-b8f6-f40598fb4705.root"

# new test on generated data, with same structure as data expected in 2024
INPUTFILE_367104="/store/group/alca_global/pps_alcareco_producer_tests/outputALCAPPSExpress.root"

# all input files and customisation commands to loop through
INPUTFILES=($INPUTFILE_355207 $INPUTFILE_367104)

# test case loop
for TEST_RUN_NO in {0..1}; do
    INPUTFILE=${INPUTFILES[$TEST_RUN_NO]}
    echo "Using file: ${INPUTFILE} , Running in: ${SCRAM_TEST_PATH} ."
    (cmsDriver.py testExpressPPSAlCaRecoProducer -s ALCAPRODUCER:PPSCalMaxTracks,ENDJOB \
        --process ALCARECO \
        --scenario pp \
        --era run3_common \
        --conditions auto:run3_data_express \
        --data  \
        --datatier ALCARECO \
        --eventcontent ALCARECO \
        --nThreads 8 \
        --number 100 --filein ${INPUTFILE} \
        --fileout file:outputALCAPPS_RECO_express_test${TEST_RUN_NO}.root \
        --customise_commands="$COMMON_CUSTOM") || die 'failed running test_express_AlCaRecoProducer' $?
done
