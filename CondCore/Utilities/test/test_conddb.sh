#!/bin/bash

function die { echo $1: status $2; exit $2; }

echo "===========> testing conddb --help"
conddb --help || die 'failed conddb --help' $?
echo -ne '\n\n'

echo "===========> testing conddb search"
conddb search 4b97f78682aac6254bbcba54cedbde468202bf5b || die 'failed conddb search' $?
echo -ne '\n\n'

echo "===========> testing conddb listParentTags"
conddb listParentTags 4b97f78682aac6254bbcba54cedbde468202bf5b || die 'failed conddb listParentTags' $?
echo -ne '\n\n'

echo "===========> testing conddb list"
conddb list SiPixelQuality_phase1_2021_v1 || die 'failed conddb list' $?
echo -ne '\n\n'

echo "===========> testing conddb copy"
conddb --yes --db pro copy SiPixelQuality_phase1_2021_v1 --destdb myfile.db  || die 'failed conddb copy' $?
echo -ne '\n\n'

echo "===========> testing conddb listTags"
conddb --db BasicPayload_v0.db listTags || die 'failed conddb listTags' $?
echo -ne '\n\n'

echo "===========> testing conddb list on local sqlite file"
conddb --db BasicPayload_v0.db list BasicPayload_v0 || die 'failed conddb list on local sqlite file' $?
echo -ne '\n\n'

echo "===========> testing conddb listGTsForTag"
conddb listGTsForTag SiPixelQuality_phase1_2021_v1  || die 'failed conddb listGTsForTag' $?
echo -ne '\n\n'

echo "===========> testing conddb diff"
conddb diff 120X_mcRun3_2021_realistic_v1 120X_mcRun3_2021_realistic_conddb_unitTest  || die 'conddb diff' $?
echo -ne '\n\n'

echo "===========> testing conddb diffGlobalTagsAtRun"
conddb diffGlobalTagsAtRun -R 120X_mcRun3_2021_realistic_v1 -T 120X_mcRun3_2021_realistic_conddb_unitTest --run 1 || die 'conddb diffGlobalTagsAtRun' $?
echo -ne '\n\n'

echo "===========> testing conddb dump"
conddb dump 4b97f78682aac6254bbcba54cedbde468202bf5b || die 'failed conddb dump payload' $?
conddb dump 4b97f78682aac6254bbcba54cedbde468202bf5b --destfile payload_dump.xml || die 'failed conddb dump payload to file' $?
conddb dump SiPixelQuality_phase1_2021_v1 || die 'failed conddb dump tag' $?
conddb dump SiPixelQuality_phase1_2021_v1 --destfile tag_dump.xml || die 'failed conddb dump tag to file' $?
conddb dump Test_CMSSW_IB_unitTest_v1 || die 'failed conddb dump gt' $?
conddb dump Test_CMSSW_IB_unitTest_v1 --destfile gt_dump.xml || die 'failed conddb dump gt to file' $?
echo -ne '\n\n'

#conddb showFCSR || die 'failed conddb showFCSR' $?  # the FCSR is not always a real run...
