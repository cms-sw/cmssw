#!/bin/bash
# exit on error
set -e

# ----- TEST CONFIGURATION

CAMPAIGN="sipixel_query_logging"

# Simulated test metadata for now
PAYLOAD_SIZE="10"
PAYLOAD_NUMBER="5"
TEST_EXECUTION="1"

# Source and destination DBs
SOURCE_DB="onlineorapro"

# Base directory for all query logging tests
BASE_TESTDIR="/data/upload_test/alejandro/15_1_0_patch2_conddb_copy_logging_test/src/CondCore/Utilities/test/conddb_query_tests"

# Timestamp for this test campaign execution
RUN_TIME="$(date +%Y-%m-%d-%Hh%Mm%S)"

# Directory for this campaign execution
TESTDIR="${BASE_TESTDIR}/${CAMPAIGN}_${RUN_TIME}"
mkdir -p "$TESTDIR"

# Test file name used for the log and DB
TESTFILE="${CAMPAIGN}_s${PAYLOAD_SIZE}_n${PAYLOAD_NUMBER}_t${TEST_EXECUTION}_${RUN_TIME}"

LOGFILE="${TESTDIR}/${TESTFILE}.log"
DEST_DB="sqlite_file:${TESTDIR}/${TESTFILE}.db"

# Final consolidated CSV for all logs in this campaign directory
CSVFILE="${TESTDIR}/${CAMPAIGN}_${RUN_TIME}_query_log.csv"

# Parser location
PARSER="${BASE_TESTDIR}/parser_query_logging.py"

# ----- SETUP CHECKS

if [ ! -f "$PARSER" ]; then
    echo "ERROR: Parser file does not exist: $PARSER"
    exit 1
fi

# If destination DB exists, ask if it should be removed
DB_FILE="${TESTDIR}/${TESTFILE}.db"

if [ -f "$DB_FILE" ]; then
    read -p "File ${DB_FILE} already exists. Do you want to remove it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$DB_FILE"
        echo "File ${DB_FILE} removed." | tee -a "$LOGFILE"
    else
        echo "File ${DB_FILE} exists and was not removed." | tee -a "$LOGFILE"
    fi
fi

# ----- EXECUTION

echo "Campaign: $CAMPAIGN"
echo "Payload size: $PAYLOAD_SIZE"
echo "Payload number: $PAYLOAD_NUMBER"
echo "Test execution: $TEST_EXECUTION"
echo "Source DB: $SOURCE_DB"
echo "Destination DB: $DEST_DB"
echo "Writing to log file: $LOGFILE"

{
    time conddb -v -a ~/ --yes \
        --db "$SOURCE_DB" \
        copy SiPixelQuality_phase1_2021_v1 \
        --destdb "$DEST_DB"
} 2>&1 | tee -a "$LOGFILE"

echo "Wrote log file: $LOGFILE"

# ----- PARSE ALL LOGS IN THIS TEST DIRECTORY

python3 "$PARSER" "$TESTDIR" \
    -o "$CSVFILE" \
    --source "$SOURCE_DB" \
    --destination "$DEST_DB"

echo "Wrote consolidated QUERY_LOG CSV file: $CSVFILE"
ls -lh "$CSVFILE"