#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}
function runFailure {
    echo "cmsRun $@ (expected to fail)"
    cmsRun $@ && die "cmsRun $*" 1
    echo
}

VERSION_ARR=(${CMSSW_VERSION//_/ })
VERSION1="${VERSION_ARR[0]}_${VERSION_ARR[1]}_${VERSION_ARR[2]}_0"
VERSION2="${VERSION_ARR[0]}_${VERSION_ARR[1]}_${VERSION_ARR[2]}_1"
VERSION3="${VERSION_ARR[0]}_${VERSION_ARR[1]}_$((${VERSION_ARR[2]}+1))_0"

# Check that changing the patch version does not lead to new lumi or run
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION1} --firstEvent 1 --output version1.dat
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION2} --firstEvent 101 --output version2.dat

CatStreamerFiles merged.dat version1.dat version2.dat

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged.dat --output merged.root

edmProvDump merged.root | grep -q "PROD.*'${VERSION1}'" || die "Did not find ${VERSION1} from merged.root provenance" $?
edmProvDump merged.root | grep -q "PROD.*'${VERSION2}'" || die "Did not find ${VERSION2} from merged.root provenance" $?


# Check that changing the minor version leads to new lumi
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION3} --firstEvent 201 --output version3_lumi.dat

CatStreamerFiles merged3_lumi.dat version1.dat version3_lumi.dat

runFailure ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged3_lumi.dat --output merged3_lumi.root

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged3_lumi.dat --output merged3_lumi.root --expectNewLumi

edmProvDump merged3_lumi.root | grep -q "PROD.*'${VERSION3}'" || die "Did not find ${VERSION3} from merged3_lumi.root provenance" $?


# Check that changing the minor version leads to new run
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION3} --firstEvent 201 --lumi 2 --output version3_run.dat

CatStreamerFiles merged3_run.dat version1.dat version3_run.dat

runFailure ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged3_run.dat --output merged3_run.root

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged3_run.dat --output merged3_run.root --expectNewRun

edmProvDump merged3_run.root | grep -q "PROD.*'${VERSION3}'" || die "Did not find ${VERSION3} from merged3_run.root provenance" $?

exit 0
