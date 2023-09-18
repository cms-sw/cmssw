#!/bin/bash

TESTDIR=${LOCALTOP}/src/FWCore/Integration/test
TEST=testModuleTypeResolver_cfg.py

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun ${TESTDIR}/${TEST} $1"
    cmsRun ${TESTDIR}/${TEST} $1 || die "cmsRun ${TEST} $1" $?
    echo
}
function runFailure {
    echo "cmsRun ${TESTDIR}/${TEST} $1 (job itself is expected to fail)"
    cmsRun -j testModuleTypeResolver_jobreport.xml ${TESTDIR}/${TEST} $1 && die "cmsRun ${TEST} $1 did not fail" 1
    EXIT_CODE=$(edmFjrDump --exitCode testModuleTypeResolver_jobreport.xml)
    if [ "x${EXIT_CODE}" != "x8035" ]; then
        echo "ModuleTypeResolver test for unavailable accelerator reported exit code ${EXIT_CODE} which is different from the expected 8035"
        exit 1
    fi
    echo
}

runSuccess ""
runSuccess "--enableOther --expectOther"
runSuccess "--enableOther --accelerators=cpu"
runSuccess "--enableOther --setInResolver=cpu"
runSuccess "--enableOther --setInResolver=other --expectOther"
runSuccess "--enableOther --setInResolver=cpu --setInModule=other --expectOther"
runSuccess "--enableOther --setInResolver=other --setInModule=cpu"
runSuccess "--enableOther --setInModule=cpu"

runFailure "--setInResolver=other --expectOther"
runFailure "--setInModule=other --expectOther"
runFailure "--setInResolver=other --setInModule=cpu"
