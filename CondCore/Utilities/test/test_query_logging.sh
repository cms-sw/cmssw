#!/bin/bash
# exit on error
set -e

# ----- ARGUMENTS

TESTNAME="sipixel_query_logging"
TEST_NR="1"

# TEST SETUP
TESTFILE="${TESTNAME}_${TEST_NR}"
     
TESTDIR=/data/upload_test/alejandro/15_1_0_patch2_conddb_copy_logging_test/src/CondCore/Utilities/test/conddb_query_tests
mkdir -p $TESTDIR

LOGFILE=$TESTDIR/${TESTFILE}_$(date +%Y-%m-%d-%Hh%Mm%S).log
DEST_DB="sqlite_file:${TESTDIR}/${TESTFILE}.db"

#if exists ask if it should be removed
if [ -f $TESTDIR/${TESTFILE}.db ]; then
    read -p "File ${TESTDIR}/${TESTFILE}.db already exists. Do you want to remove it? (y/n) " -n 1 -r
    echo    # move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm $TESTDIR/${TESTFILE}.db
        echo "File ${TESTDIR}/${TESTFILE}.db removed." | tee -a $LOGFILE
    else
        echo "File ${TESTDIR}/${TESTFILE}.db exists and was not removed." | tee -a $LOGFILE
    fi
fi

# ------- EXECUTION
echo "writing to log file: $LOGFILE"

{ time 
conddb -v -a ~/ --yes --db onlineorapro copy SiPixelQuality_phase1_2021_v1 --destdb $DEST_DB;
 } 2>&1 | tee -a $LOGFILE

echo "wrote to log file: $LOGFILE"

CSVFILE="${LOGFILE%.log}_query_log.csv"

python3 $TESTDIR/parser_query_logging.py "$LOGFILE" -o "$CSVFILE"

echo "wrote QUERY_LOG CSV file: $CSVFILE"
ls -lh "$CSVFILE"