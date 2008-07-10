#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/gen_things_cfg.py || die 'Failed generating test root file' $?

root -b -n -q ${LOCAL_TEST_DIR}/thing_sel.C || die 'Failed tfwliteselectortest::ThingsTSelector test' $?
root -b -n -q ${LOCAL_TEST_DIR}/thing2_sel.C || die 'Failed tfwliteselectortest::ThingsTSelector2 test' $?
