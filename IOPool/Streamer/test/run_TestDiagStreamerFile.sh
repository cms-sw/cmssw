#!/bin/bash

function die { echo -e "Failure $1: status $2" ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}

VERSION_ARR=(${CMSSW_VERSION//_/ })
VERSION1="${VERSION_ARR[0]}_${VERSION_ARR[1]}_${VERSION_ARR[2]}_0"
VERSION2="${VERSION_ARR[0]}_${VERSION_ARR[1]}_${VERSION_ARR[2]}_1"
VERSION3="${VERSION_ARR[0]}_${VERSION_ARR[1]}_$((${VERSION_ARR[2]}+1))_0"

# Check compression algorithms
for COMP in "UNCOMPRESSED" "ZLIB" "LZMA" "ZSTD"; do
    runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --compression ${COMP} --output compression_${COMP}.dat
    DiagStreamerFile compression_${COMP}.dat > compression_${COMP}_diag.txt || die "DiagStreamerFile compression_${COMP}.dat failed, last 10 lines:\n $(tail -n 10 compression_${COMP}_diag.txt)" $?
    grep -q "and 10 events" compression_${COMP}_diag.txt || die "compression_${COMP}_diag.txt had incorrect event count" $?
done


# Check that changing the patch version does not lead to new lumi or run
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION1} --firstEvent 1 --output version1.dat
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION2} --firstEvent 101 --output version2.dat

CatStreamerFiles merged.dat version1.dat version2.dat
DiagStreamerFile merged.dat > merged_diag.txt 2>&1 || die "DiagStreamerFile merged.dat failed, last 10 lines:\n $(tail -n 10 merged_diag.txt)" $?
grep -q "read 2 metadata records" merged_diag.txt || die "merged_diag.txt had incorrect metadata record count" $?
grep -q "and 20 events" merged_diag.txt || die "merged_diag.txt had incorrect event count" $?
grep -q "and 0 events with bad headers" merged_diag.txt || die "merged_diag.txt had incorrect bad headers count" $?
grep -q "and 0 events with bad check sum" merged_diag.txt || die "merged_diag.txt had incorrect bad check sum count" $?
grep -q "and 0 duplicated event Id" merged_diag.txt || die "merged_diag.txt had incorrect duplicated event count" $?

# Check that changing the minor version leads to new lumi
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --version ${VERSION3} --firstEvent 201 --output version3.dat

CatStreamerFiles merged3.dat version1.dat version3.dat
DiagStreamerFile merged3.dat > merged3_diag.txt 2>&1 || die "DiagStreamerFile merged3.dat failed, last 10 lines:\n $(tail -n 10 merged3_diag.txt)" $?
grep -q "ProcessHistoryID error for count 11" merged3_diag.txt || die "merged3_diag.txt did not have ProcessHistoryID error for count 11" $?
grep -q "ProcessHistoryID error for count 20" merged3_diag.txt || die "merged3_diag.txt did not have ProcessHistoryID error for count 20" $?
NUM=$(grep -c "ProcessHistoryID error" merged3_diag.txt)
if [ "$NUM" != "10" ]; then
    die "merged3_diag.txt had unexpected number $NUM of ProcessHistoryID errors, expected 10" 1
fi
grep -q "and 10 events with incompatible reduced ProcessHistoryID" merged3_diag.txt || die "merged3_diag.txt had incorrect 'incompatible reduced ProcessHistoryID' count" $?
