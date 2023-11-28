#!/bin/bash
function die { echo $1: status $2 ;  exit $2; }
run="319176"
dqmFile="/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2018/ZeroBias/R0003191xx/DQM_V0001_R000319176__ZeroBias__Run2018B-PromptReco-v2__DQMIO.root"
if [ ! -f "${dqmFile}" ]; then
  die "SKIPPING test, file ${dqmFile} not found" 0
fi
runStartTime=$( (python "${SCRAM_TEST_PATH}/cfg/getRunStartTime.py" "${run}" | tail -n1 ) || die "Failed to get run start time" $? )
echo "DEBUG: Run ${run} started at ${runStartTime}"
(cmsRun "${SCRAM_TEST_PATH}/cfg/makeMergeBadComponentPayload_example_cfg.py" globalTag=auto:run3_data_prompt runNumber="${run}"  runStartTime="${runStartTime}" dqmFile="${dqmFile}" dbfile="test.db" outputTag="TestBadComponents" ) || die "Failure running cmsRun makeMergeBadComponentPayload_example_cfg.py" $?
# get the hash
plEntryLn=$( (conddb --db sqlite_file:test.db list TestBadComponents | grep SiStripBadStrip) || die "Failed to get payload" $? )
read -a plEntryLn_split <<< "${plEntryLn}"
plHash="${plEntryLn_split[-2]}"
(conddb --db sqlite_file:test.db dump "${plHash}" --format=xml -d test_dump.log ) || die "Could not dump test payload" $?
(conddb dump 750b6c7049ccd4d2448944b22c2d4d16bf9cc30c --format=xml -d ref_dump.log ) || die "Could not dump reference payload" $?  # from SiStripBadComponents_2018_forSpecialBetaRun_v0_mc
(diff -u ref_dump.log test_dump.log ) || die "Different payloads" $?
