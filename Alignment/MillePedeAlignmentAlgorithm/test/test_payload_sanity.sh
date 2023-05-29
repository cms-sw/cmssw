#!/bin/bash
function die { echo $1: status $2; exit $2; }

INPUTFILE=${SCRAM_TEST_PATH}/alignments_MP.db
(cmsRun ${SCRAM_TEST_PATH}/AlignmentRcdChecker_cfg.py inputSqliteFile=${INPUTFILE}) || die 'failed running AlignmentRcdChecker'
rm $INPUTFILE
