#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_L1Scouting_test_file_cfg.py || die 'Failure using create_L1Scouting_test_file_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py || die "Failure using read_L1Scouting_cfg.py" $?

# The input files used below were generated
# using the CMSSW release specified in their names.
#
# For every release in question, two files were created,
# using either splitLevel=0 or splitLevel=99
# in the configuration of the EDM output module.
# More info on the rationale to test both split levels can be found in
# https://github.com/cms-sw/cmssw/issues/45931
#
# Below, an example command to produce such files
# (it assumes the existence of some directories).
#
#  for splitLevel in 0 99; do
#    cmsRun DataFormats/L1Scouting/test/create_L1Scouting_test_file_cfg.py \
#      -o DataFormats/L1Scouting/data/testL1Scouting_vA_vB_vC_vD_vE_vF_vG_vH_"${CMSSW_VERSION:6}"_split_"${splitLevel}".root \
#      -s "${splitLevel}"
#  done
#
# Every pair of files coincides with
# (a) the introduction of new L1-Scouting data formats, and/or
# (b) the update of some of the existing data formats.
# The versioning of newly-introduced data formats starts at "v3".
#
# The version number of the L1-Scouting data formats is specified
# in the names of the input files, according to the following order.
#
#  l1ScoutingRun3::Muon
#  l1ScoutingRun3::EGamma
#  l1ScoutingRun3::Tau
#  l1ScoutingRun3::Jet
#  l1ScoutingRun3::BxSums
#  l1ScoutingRun3::BMTFStub
#  l1ScoutingRun3::CaloTower
#  l1ScoutingRun3::CaloJet
#
# For files produced before the introduction of a given class,
# only the versions of the previous classes are included in the name of the files.

# test file for muon, jet, e/gamma, tau and energy sums data formats
oldFiles="testL1Scouting_v3_v3_v3_v3_v3_14_0_0_split_99.root testL1Scouting_v3_v3_v3_v3_v3_14_0_0_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$file) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$file" $?
  cmsRunArgs="-i ${inputfile}"
  cmsRunArgs+=" --bmtfStubVersion 0 --caloTowerVersion 0 --caloJetVersion 0"
  cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py ${cmsRunArgs} || die "Failed to read old file $file" $?
done

# added BMTF input stubs data format
oldFiles="testL1Scouting_v3_v3_v3_v3_v3_v3_14_1_0_pre5_split_99.root testL1Scouting_v3_v3_v3_v3_v3_v3_14_1_0_pre5_split_0.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$file) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$file" $?
  cmsRunArgs="-i ${inputfile}"
  cmsRunArgs+=" --bmtfStubVersion 3 --caloTowerVersion 0 --caloJetVersion 0"
  cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py ${cmsRunArgs} || die "Failed to read old file $file" $?
done

# added CaloTower and CaloJet data formats
oldFiles="testL1Scouting_v3_v3_v3_v3_v3_v3_v3_v3_16_1_0_pre3_split_0.root testL1Scouting_v3_v3_v3_v3_v3_v3_v3_v3_16_1_0_pre3_split_99.root"
for file in $oldFiles; do
  inputfile=$(edmFileInPath DataFormats/L1Scouting/data/$file) || die "Failure edmFileInPath DataFormats/L1Scouting/data/$file" $?
  cmsRunArgs="-i ${inputfile}"
  cmsRunArgs+=" --bmtfStubVersion 3 --caloTowerVersion 3 --caloJetVersion 3"
  cmsRun ${LOCAL_TEST_DIR}/read_L1Scouting_cfg.py ${cmsRunArgs} || die "Failed to read old file $file" $?
done
