#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }
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
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION1} --firstEvent 1 --output version1.root
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION2} --firstEvent 101 --output version2.root

edmProvDump version1.root | grep -q "PROD.*'${VERSION1}'" || die "Did not find ${VERSION1} from version.root provenance" $?
edmProvDump version2.root | grep -q "PROD.*'${VERSION2}'" || die "Did not find ${VERSION2} from version.root provenance" $?

runSuccess ${SCRAM_TEST_PATH}/test_merge_two_files.py version1.root version2.root

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged_files.root


# Check that changing the minor version leads to new lumi
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION3} --firstEvent 201 --output version3_lumi.root

edmProvDump version3_lumi.root | grep -q "PROD.*'${VERSION3}'" || die "Did not find ${VERSION3} from version3_lumi.root provenance" $?

runSuccess ${SCRAM_TEST_PATH}/test_merge_two_files.py version1.root version3_lumi.root --output merged_files3_lumi.root --bypassVersionCheck

runFailure ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged_files3_lumi.root --bypassVersionCheck

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged_files3_lumi.root --bypassVersionCheck --expectNewLumi


# Check that changing the minor version leads to new run
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION3} --firstEvent 201 --lumi 2 --output version3_run.root

edmProvDump version3_run.root | grep -q "PROD.*'${VERSION3}'" || die "Did not find ${VERSION3} from version3_lumi.root provenance" $?

runSuccess ${SCRAM_TEST_PATH}/test_merge_two_files.py version1.root version3_run.root --output merged_files3_run.root --bypassVersionCheck

runFailure ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged_files3_run.root --bypassVersionCheck

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged_files3_run.root --bypassVersionCheck --expectNewRun

exit 0
