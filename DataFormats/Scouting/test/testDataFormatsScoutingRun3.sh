#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  if [ $# -gt 2 ]; then
    printf "%s\n" "=== Log File =========="
    cat $3
    printf "%s\n" "=== End of Log File ==="
  fi
  exit $2
}

# read Scouting collections from existing EDM file, and write them to disk
cmsRun "${SCRAM_TEST_PATH}"/scoutingCollectionsIO_cfg.py -- \
  -i /store/mc/Run3Summer22DR/GluGlutoHHto2B2Tau_kl-5p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/GEN-SIM-RAW/124X_mcRun3_2022_realistic_v12-v2/2550000/bbfb86f3-4073-47e3-967b-059aa6b904ad.root \
  -n 150 --skip 0 -o testDataFormatsScoutingRun3_step1.root \
  >testDataFormatsScoutingRun3_step1.log 2>testDataFormatsScoutingRun3_step1_stderr.log \
  || die "Failure running scoutingCollectionsIO_cfg.py" $? testDataFormatsScoutingRun3_step1_stderr.log

cat testDataFormatsScoutingRun3_step1.log

# validate content of Scouting collections
"${SCRAM_TEST_PATH}"/scoutingCollectionsDumper.py -v 1 -n 1 --skip 137 -i testDataFormatsScoutingRun3_step1.root -k Run3Scouting \
  > testDataFormatsScoutingRun3_step2.log 2>testDataFormatsScoutingRun3_step2_stderr.log \
  || die "Failure running scoutingCollectionsDumper.py" $? testDataFormatsScoutingRun3_step2_stderr.log

diff -q "${SCRAM_TEST_PATH}"/testDataFormatsScoutingRun3_expected.log testDataFormatsScoutingRun3_step2.log \
  || die "Unexpected differences in outputs of testDataFormatsScoutingRun3 (step 2)" $?
