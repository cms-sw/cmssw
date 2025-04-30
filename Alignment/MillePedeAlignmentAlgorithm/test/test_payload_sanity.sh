#!/bin/bash
function die { echo $1: status $2; exit $2; }

echo -e "Content of the current directory is: "`ls .`
INPUTFILE=alignments_MP.db

# Check if the file exists
if [ ! -f "$INPUTFILE" ]; then
    echo "Error: $INPUTFILE does not exist."
    exit 1
fi

(cmsRun ${SCRAM_TEST_PATH}/AlignmentRcdChecker_cfg.py inputSqliteFile=${INPUTFILE}) || die 'failed running AlignmentRcdChecker' $?
rm $INPUTFILE
-- dummy change --
