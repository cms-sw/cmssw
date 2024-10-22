#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

CONFIG=${LOCALTOP}/src/FWCore/Framework/test/test_conditionaltasks_nonconsumed_cfg.py
OUTPUT=conditionaltasks_nonconsumed.log
REFERENCE=${LOCALTOP}/src/FWCore/Framework/test/unit_test_outputs/$OUTPUT

function run {
    cmsRun $CONFIG $@
    tail -n +2 conditionaltasks_nonconsumed.log | diff - $REFERENCE || die "cmsRun $CONFIG $@ provides unexpected log"
}

run
run --filterSucceeds
run --testView
