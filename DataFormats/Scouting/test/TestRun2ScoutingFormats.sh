#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_Run2Scouting_test_file_cfg.py || die 'Failure using create_Run2Scouting_test_file_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_readRun2Scouting_cfg.py || die "Failure using test_readRun2Scouting_cfg.py" $?

# The old files read below were generated as follows.
#
#     testRun2Scouting_v3_v2_v2_v3_v2_v2_v2_CMSSW_8_0_7.root:
#     Check out the 8_0_7 release and cherry pick the commit that
#     adds the original version of the file
#     DataFormats/Scouting/test/TestWriteRun2Scouting.cc
#     (6 files added by that commit). There likely will be
#     minor conflicts or issues in test/BuildFile.xml that need to
#     be resolved.
#
#     testRun2Scouting_v3_v2_v3_v3_v2_v2_v3_CMSSW_9_4_0.root
#     Check out the 9_4_0 release and cherry pick the commit that
#     adds the original version of the file
#     DataFormats/Scouting/test/TestWriteRun2Scouting.cc.
#     Also check out the second commit that modifies that file.
#     (6 files added by those commits). There likely will be
#     minor conflicts or issues in test/BuildFile.xml that need to
#     be resolved.
#
#     testRun2Scouting_v3_v2_v3_v3_v2_v2_v2_v3_CMSSW_10_2_0.root
#     Check out the 10_2_0 release and cherry pick the commit that
#     adds the original version of the file
#     DataFormats/Scouting/test/TestWriteRun2Scouting.cc.
#     Also check out the second and third commits that modify that file.
#     (6 files added by those commits). There likely will be
#     minor conflicts or issues in test/BuildFile.xml that need to
#     be resolved.
#
# Run the create_Run2Scouting_test_file_cfg.py configuration and
# rename the file it creates.
#
# The versions of the classes are encoded in the filenames in
# alphabetical order. This order is also the order the Run 2
# Scouting files appear in classes_def.xml (in the master branch
# of CMSSW).
#
# The versions that changed from 8_0_7 to 9_4_0 were the versions of
# the ScoutingMuon and ScoutingVertex classes. The change in the
# 10_2_0 was the addition of the ScoutingTrack class.
#
# 8_0_7 was chosen because the 8_0_X release cycle was used
# for 2016 data, and the scouting data formats were not updated
# after 8_0_7.
#
# 9_4_0 was chosen because the 9_4_X release cycle was used
# for 2017 data, and the scouting data formats were not updated
# after 9_4_0.
#
# 10_2_0 was chosen because the 10_2_X release cycle was used
# for 2018 data, and the scouting data formats were not updated
# after 10_2_0.
#

file=testRun2Scouting_v3_v2_v2_v3_v2_v2_v2_CMSSW_8_0_7.root
inputfile=$(edmFileInPath DataFormats/Scouting/data/$file) || die "Failure edmFileInPath DataFormats/Scouting/data/$file" $?
argsPassedToPython="-- --inputFile $inputfile --outputFileName testRun2Scouting2_CMSSW_8_0_7.root --muonVersion 2 --trackVersion 0 --vertexVersion 2"
cmsRun ${LOCAL_TEST_DIR}/test_readRun2Scouting_cfg.py $argsPassedToPython || die "Failed to read old file $file" $?

file=testRun2Scouting_v3_v2_v3_v3_v2_v2_v3_CMSSW_9_4_0.root
inputfile=$(edmFileInPath DataFormats/Scouting/data/$file) || die "Failure edmFileInPath DataFormats/Scouting/data/$file" $?
argsPassedToPython="-- --inputFile $inputfile --outputFileName testRun2Scouting2_CMSSW_9_4_0.root --muonVersion 3 --trackVersion 0 --vertexVersion 3"
cmsRun ${LOCAL_TEST_DIR}/test_readRun2Scouting_cfg.py  $argsPassedToPython || die "Failed to read old file $file" $?

file=testRun2Scouting_v3_v2_v3_v3_v2_v2_v2_v3_CMSSW_10_2_0.root
inputfile=$(edmFileInPath DataFormats/Scouting/data/$file) || die "Failure edmFileInPath DataFormats/Scouting/data/$file" $?
argsPassedToPython="-- --inputFile $inputfile --outputFileName testRun2Scouting2_CMSSW_10_2_0.root --muonVersion 3 --trackVersion 2 --vertexVersion 3"
cmsRun ${LOCAL_TEST_DIR}/test_readRun2Scouting_cfg.py $argsPassedToPython || die "Failed to read old file $file" $?

exit 0
