#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_Run3Scouting_test_file_cfg.py || die 'Failure using create_Run3Scouting_test_file_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_readRun3Scouting_cfg.py || die "Failure using test_readRun3Scouting_cfg.py" $?

# The old files read below were generated as follows.
#
#     testRun3Scouting_v3_v5_v3_v4_v5_v3_v5_v3_v3_CMSSW_12_4_0.root:
#     Check out the 12_4_0 release and cherry pick the commit that
#     adds the original version of the file
#     DataFormats/Scouting/test/TestWriteRun3Scouting.cc
#     (6 files added by that commit). There likely will be
#     minor conflicts or issues in test/BuildFile.xml that need to
#     be resolved.
#
#     testRun3Scouting_v3_v6_v3_v4_v5_v3_v5_v3_v3_CMSSW_13_0_3.root:
#     Check out the 13_0_3 release and cherry pick the commit that
#     adds the original version of the file
#     DataFormats/Scouting/test/TestWriteRun3Scouting.cc
#     Also check out the second commit that modifies that file.
#     (6 files added by those commits). There likely will be
#     minor conflicts or issues in test/BuildFile.xml that need to
#     be resolved.
#
# Run the create_Run3Scouting_test_file_cfg.py configuration and
# rename the file it creates.
#
# The versions of the classes are encoded in the filenames in
# alphabetical order. This order is also the order the Run 3
# Scouting files appear in classes_def.xml (in the master branch
# of CMSSW).
#
# The only version that changed from 12_4_0
# to 13_0_3 was the Run3ScoutingElectron class.
#
# 12_4_0 was chosen because the 12_4_X release cycle was used
# for 2022 data, and the scouting data formats were not updated
# after 12_4_0.
#
# 13_0_3 was chosen because the 13_0_X release cycle was used
# for 2023 data, and the scouting data formats and ROOT were
# not updated after 13_0_3.

file=testRun3Scouting_v3_v5_v3_v4_v5_v3_v5_v3_v3_CMSSW_12_4_0.root
inputfile=$(edmFileInPath DataFormats/Scouting/data/$file) || die "Failure edmFileInPath DataFormats/Scouting/data/$file" $?
argsPassedToPython="-- --inputFile $inputfile --outputFileName testRun3Scouting2_CMSSW_12_4_0.root --electronVersion 5"
cmsRun ${LOCAL_TEST_DIR}/test_readRun3Scouting_cfg.py $argsPassedToPython || die "Failed to read old file $file" $?

file=testRun3Scouting_v3_v6_v3_v4_v5_v3_v5_v3_v3_CMSSW_13_0_3.root
inputfile=$(edmFileInPath DataFormats/Scouting/data/$file) || die "Failure edmFileInPath DataFormats/Scouting/data/$file" $?
argsPassedToPython="-- --inputFile $inputfile --outputFileName testRun3Scouting2_CMSSW_13_0_3.root --electronVersion 6"
cmsRun ${LOCAL_TEST_DIR}/test_readRun3Scouting_cfg.py $argsPassedToPython || die "Failed to read old file $file" $?

exit 0
