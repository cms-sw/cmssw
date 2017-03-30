#!/bin/sh

TEST_DIR="/tmp/MuonGeomtryDBConverter_$(date '+%G-%m-%d_%H.%M.%S.%N')_${RANDOM}"
TEST_CFG="${CMSSW_BASE}/src/Alignment/MuonAlignment/test/muonGeometryDBConverter_cfg.py"

clean_up() {
    rm -rf ${TEST_DIR}
}
trap clean_up EXIT

check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}


rm -rf ${TEST_DIR}
mkdir -p ${TEST_DIR}

check_for_success cmsRun ${TEST_CFG} input=ideal output=db outputFile=${TEST_DIR}/ideal.db
check_for_success cmsRun ${TEST_CFG} input=db output=xml inputFile=${TEST_DIR}/ideal.db outputFile=${TEST_DIR}/ideal.xml
check_for_success cmsRun ${TEST_CFG} input=xml output=none inputFile=${TEST_DIR}/ideal.xml

clean_up
