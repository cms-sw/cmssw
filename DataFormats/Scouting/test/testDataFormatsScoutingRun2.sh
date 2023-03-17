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
  -i /store/mc/RunIISummer20UL18RECO/DoubleElectron_Pt-1To300-gun/AODSIM/FlatPU0to70EdalIdealGT_EdalIdealGT_106X_upgrade2018_realistic_v11_L1v1_EcalIdealIC-v2/270000/4CDD9457-E14C-D84A-9BD4-3140CB6AEEB6.root \
  -n 150 --skip 900 -o testDataFormatsScoutingRun2_step1.root &> testDataFormatsScoutingRun2_step1.log \
  || die "Failure running scoutingCollectionsIO_cfg.py" $? testDataFormatsScoutingRun2_step1.log

cat testDataFormatsScoutingRun2_step1.log

# validate content of Scouting collections
"${SCRAM_TEST_PATH}"/scoutingCollectionsDumper.py -v 1 -n 1 --skip 81 -i testDataFormatsScoutingRun2_step1.root -k Scouting \
  &> testDataFormatsScoutingRun2_step2.log \
  || die "Failure running scoutingCollectionsDumper.py" $? testDataFormatsScoutingRun2_step2.log

diff -q "${SCRAM_TEST_PATH}"/testDataFormatsScoutingRun2_expected.log testDataFormatsScoutingRun2_step2.log \
  || die "Unexpected differences in outputs of testDataFormatsScoutingRun2 (step 2)" $?
