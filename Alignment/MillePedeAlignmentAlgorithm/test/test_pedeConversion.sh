#!/bin/bash
function die { echo $1: status $2; exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

echo "============== testing conversion to DB file from millepede.res"
(cmsRun ${LOCAL_TEST_DIR}/convertMPresToDB_cfg.py) || die 'failed running convertMPresToDB_cfg.py' $?

echo -e "Content of the current directory is: "`ls .`

INPUTFILE=convertedFromResFile.db

echo -e "\n\n============== testing converted file corresponds to input"
(cmsRun ${SCRAM_TEST_PATH}/AlignmentRcdChecker_cfg.py inputSqliteFile=${INPUTFILE}) || die 'failed running AlignmentRcdChecker' $?
-- dummy change --
