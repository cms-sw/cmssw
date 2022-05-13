#!/bin/bash
function die { echo $1: status $2; exit $2; }

INPUTFILE=${LOCAL_TEST_DIR}/alignments_MP.db
(cmsRun ${LOCAL_TEST_DIR}/AlignmentRcdChecker_cfg.py inputSqliteFile=${INPUTFILE}) || die 'failed running AlignmentRcdChecker'
rm $INPUTFILE
