#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
export TMPDIR=`pwd`
cmsRun ${LOCAL_TEST_DIR}/gen_things_cfg.py || die 'Failed generating test root file' $?

rm -f a.jpg refA.jpg
root -b -n -q ${LOCAL_TEST_DIR}/thing_sel.C || die 'Failed tfwliteselectortest::ThingsTSelector test' $?
[ -s a.jpg ] && [ -s refA.jpg ] || die 'Failed tfwliteselectortest::ThingsTSelector test, no histograms' 20

rm -f a.jpg refA.jpg
root -b -n -q ${LOCAL_TEST_DIR}/thing2_sel.C || die 'Failed tfwliteselectortest::ThingsTSelector2 test' $?
[ -s a.jpg ] && [ -s refA.jpg ] || die 'Failed tfwliteselectortest::ThingsTSelector2 test, no histograms' 20

rm -f a.jpg refA.jpg
root -b -n -q ${LOCAL_TEST_DIR}/proof_thing_sel.C || die 'Failed tfwliteselectortest::ThingsTSelector test' $?
[ -s a.jpg ] && [ -s refA.jpg ] || die 'Failed tfwliteselectortest::ThingsTSelector test, no histograms' 20

rm -f a.jpg refA.jpg
root -b -n -q ${LOCAL_TEST_DIR}/proof_thing2_sel.C || die 'Failed tfwliteselectortest::ThingsTSelector2 test' $?
[ -s a.jpg ] && [ -s refA.jpg ] || die 'Failed tfwliteselectortest::ThingsTSelector2 test, no histograms' 20

rm -f a.jpg refA.jpg

exit 0
