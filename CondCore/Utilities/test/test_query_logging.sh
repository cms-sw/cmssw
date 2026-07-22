#!/bin/bash
set -euo pipefail

# ----------------------------
# DEFAULT CONFIGURATION
# ----------------------------

CMSSW_PATH="/data/upload_test/alejandro/15_1_0_patch2_conddb_copy_logging_test/src"

BASE_TESTSDIR="${CMSSW_PATH}/CondCore/Utilities/test/conddb_query_tests"
PARSER="${BASE_TESTSDIR}/parser_query_logging.py"
PAYLOADSIMFILE="${CMSSW_PATH}/CondTools/RunInfo/test/LHCInfoPerFillWriter_cfg.py"

CAMPAIGN="sipixel_query_logging"

CREATE_PAYLOADS="false"

SOURCE_DB=""
DEST_DB=""
TAG=""

PAYLOAD_SIZE="10"
PAYLOAD_NUMBER="1"
TEST_EXECUTIONS="1"

REMOVE_FAKE_DBS="true"

RUN_TIME="$(date +%Y-%m-%d-%Hh%Mm%S)"

# ----------------------------
# HELP MESSAGE
# ----------------------------

usage() {
    cat <<EOF
Usage:

  Real source DB mode:
    $0 --source-db onlineorapro --tag SiPixelQuality_phase1_2021_v1 --executions 3

  Fake payload mode:
    $0 --create-payloads --payload-size 5 --payload-number 10 --executions 3

Options:

  --create-payloads
      Create fake source DBs using cmsRun before each test execution.

  --source-db DB
      Source DB for conddb copy.
      Required when not using --create-payloads.

  --dest-db DB
      Destination DB.
      Optional. If not provided, one sqlite destination DB is created per execution.

  --tag TAG
      Tag to copy.
      Required for real source DB mode.
      Also used in fake payload mode for conddb copy.

  --payload-size N
      Payload size passed to cmsRun in fake payload mode.
      Default: ${PAYLOAD_SIZE}

  --payload-number N
      Number of payloads passed to cmsRun in fake payload mode.
      Default: ${PAYLOAD_NUMBER}

  --executions N
      Number of times to run the test.
      Default: ${TEST_EXECUTIONS}

  --campaign NAME
      Campaign name.
      Default: ${CAMPAIGN}

  --cmssw-path PATH
      CMSSW src path.
      Default: ${CMSSW_PATH}

  --keep-fake-dbs
      Do not delete fake source DBs at the end.

  -h, --help
      Show this help message.

EOF
}

# ----------------------------
# ARGUMENT PARSING
# ----------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --create-payloads)
            CREATE_PAYLOADS="true"
            shift
            ;;
        --source-db)
            SOURCE_DB="$2"
            shift 2
            ;;
        --dest-db)
            DEST_DB="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --payload-size)
            PAYLOAD_SIZE="$2"
            shift 2
            ;;
        --payload-number)
            PAYLOAD_NUMBER="$2"
            shift 2
            ;;
        --executions)
            TEST_EXECUTIONS="$2"
            shift 2
            ;;
        --campaign)
            CAMPAIGN="$2"
            shift 2
            ;;
        --cmssw-path)
            CMSSW_PATH="$2"
            BASE_TESTSDIR="${CMSSW_PATH}/CondCore/Utilities/test/conddb_query_tests"
            PARSER="${BASE_TESTSDIR}/parser_query_logging.py"
            PAYLOADSIMFILE="${CMSSW_PATH}/CondCore/RunInfo/test/LHCInfoPerFillWriter_cfg.py"
            shift 2
            ;;
        --keep-fake-dbs)
            REMOVE_FAKE_DBS="false"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# ----------------------------
# VALIDATION
# ----------------------------

if [ ! -f "$PARSER" ]; then
    echo "ERROR: Parser file does not exist: $PARSER"
    exit 1
fi

if [ "$CREATE_PAYLOADS" = "true" ]; then
    if [ ! -f "$PAYLOADSIMFILE" ]; then
        echo "ERROR: Payload simulation file does not exist: $PAYLOADSIMFILE"
        exit 1
    fi
else
    if [ -z "$SOURCE_DB" ]; then
        echo "ERROR: --source-db is required when not using --create-payloads"
        exit 1
    fi
fi

if ! [[ "$TEST_EXECUTIONS" =~ ^[0-9]+$ ]] || [ "$TEST_EXECUTIONS" -lt 1 ]; then
    echo "ERROR: --executions must be a positive integer"
    exit 1
fi

# ----------------------------
# DIRECTORY SETUP
# ----------------------------

if [ "$CREATE_PAYLOADS" = "true" ]; then
    TEST_MODE="fake_payloads"
    TEST_METADATA="s${PAYLOAD_SIZE}_n${PAYLOAD_NUMBER}_exec${TEST_EXECUTIONS}"
else
    TEST_MODE="source_db"

    SAFE_SOURCE_DB="$(echo "$SOURCE_DB" | sed 's#[/:]#_#g')"
    SAFE_TAG="$(echo "$TAG" | sed 's#[/:]#_#g')"

    TEST_METADATA="${SAFE_SOURCE_DB}_${SAFE_TAG}_exec${TEST_EXECUTIONS}"
fi

TEST_NAME="${CAMPAIGN}_${RUN_TIME}_${TEST_METADATA}_${TEST_MODE}"

TESTDIR="${BASE_TESTSDIR}/${TEST_NAME}"
mkdir -p "$TESTDIR"

CSVFILE="${TESTDIR}/${TEST_NAME}_query_log.csv"

FAKE_DBS_TO_REMOVE=()

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

make_dest_db() {
    local execution="$1"

    if [ -n "$DEST_DB" ]; then
        # If user includes {run}, replace it with the execution number.
        if [[ "$DEST_DB" == *"{run}"* ]]; then
            echo "${DEST_DB//\{run\}/${execution}}"
        else
            # If user gave a fixed destination DB and there are multiple executions,
            # make it unique to avoid collisions.
            if [ "$TEST_EXECUTIONS" -gt 1 ]; then
                if [[ "$DEST_DB" == sqlite_file:* ]]; then
                    local path="${DEST_DB#sqlite_file:}"
                    local base="${path%.db}"
                    echo "sqlite_file:${base}_run${execution}.db"
                else
                    echo "${DEST_DB}"
                fi
            else
                echo "$DEST_DB"
            fi
        fi
    else
        echo "sqlite_file:${TESTDIR}/${CAMPAIGN}_dest_run${execution}.db"
    fi
}

make_fake_source_db() {
    local execution="$1"
    echo "${TESTDIR}/${CAMPAIGN}_fake_source_s${PAYLOAD_SIZE}_n${PAYLOAD_NUMBER}_run${execution}.db"
}

run_parser() {
    local parser_source="$1"
    local parser_destination="$2"

    python3 "$PARSER" "$TESTDIR" \
        -o "$CSVFILE" \
        --source "$parser_source" \
        --destination "$parser_destination"
}

# ----------------------------
# TEST EXECUTION LOOP
# ----------------------------

for execution in $(seq 1 "$TEST_EXECUTIONS"); do
    echo
    echo "============================================================"
    echo "Starting test execution ${execution}/${TEST_EXECUTIONS}"
    echo "============================================================"

    CAMPAIGNFILEPATH="${CAMPAIGN}_s${PAYLOAD_SIZE}_n${PAYLOAD_NUMBER}_t${execution}_${RUN_TIME}"
    LOGFILE="${TESTDIR}/${CAMPAIGNFILEPATH}.log"

    RUN_SOURCE_DB="$SOURCE_DB"

    if [ "$CREATE_PAYLOADS" = "true" ]; then
        FAKE_DB_FILE="$(make_fake_source_db "$execution")"
        FAKE_DBS_TO_REMOVE+=("$FAKE_DB_FILE")

        rm -f "$FAKE_DB_FILE"

        # TODO: Add payload number
        cmsRun "$PAYLOADSIMFILE" \
            size="$PAYLOAD_SIZE" \
            db="sqlite_file:${FAKE_DB_FILE}" 

        RUN_SOURCE_DB="sqlite:${FAKE_DB_FILE}"

        echo "Created fake source DB: $RUN_SOURCE_DB" | tee -a "$LOGFILE"
    fi

    RUN_DEST_DB="$(make_dest_db "$execution")"

    if [[ "$RUN_DEST_DB" == sqlite_file:* ]]; then
        DEST_DB_FILE="${RUN_DEST_DB#sqlite_file:}"
        rm -f "$DEST_DB_FILE"
    fi

    echo "Source DB: $RUN_SOURCE_DB" | tee -a "$LOGFILE"
    echo "Dest DB:   $RUN_DEST_DB" | tee -a "$LOGFILE"

    PARSER_SOURCE_DB="$RUN_SOURCE_DB"
    PARSER_DEST_DB="$RUN_DEST_DB"

    if [ "$CREATE_PAYLOADS" = "true" ]; then
        PARSER_SOURCE_DB="sqlite:$(basename "$FAKE_DB_FILE")"
    fi

    if [[ "$RUN_DEST_DB" == sqlite:* ]]; then
        DEST_DB_PATH="${RUN_DEST_DB#sqlite_file:}"
        PARSER_DEST_DB="sqlite:$(basename "$DEST_DB_PATH")"
    fi

    if [[ "$RUN_DEST_DB" == sqlite_file:* ]]; then
        DEST_DB_PATH="${RUN_DEST_DB#sqlite:}"
        PARSER_DEST_DB="sqlite:$(basename "$DEST_DB_PATH")"
    fi

    if [ "$CREATE_PAYLOADS" = "true" ]; then
        {
            time conddb -v -a ~/ --yes --force \
                --db "$RUN_SOURCE_DB" \
                copy "LHCInfoPerFillFake" "mocktest" \
                --note "Mock test Query time DB" \
                --destdb "$RUN_DEST_DB"
        } 2>&1 | tee -a "$LOGFILE"
    else
        {
            time conddb -v -a ~/ --yes \
                --db "$RUN_SOURCE_DB" \
                copy "$TAG" \
                --destdb "$RUN_DEST_DB"
        } 2>&1 | tee -a "$LOGFILE"
    fi

    echo "Wrote log file: $LOGFILE"

    echo "Parsing logs after execution $execution..."
    run_parser "$PARSER_SOURCE_DB" "$PARSER_DEST_DB"

    echo "Updated CSV file: $CSVFILE"
done

# ----------------------------
# CLEANUP FAKE DBS
# ----------------------------

if [ "$CREATE_PAYLOADS" = "true" ] && [ "$REMOVE_FAKE_DBS" = "true" ]; then
    echo
    echo "Removing fake source DBs..."

    for fake_db in "${FAKE_DBS_TO_REMOVE[@]}"; do
        if [ -f "$fake_db" ]; then
            rm -f "$fake_db"
            echo "Removed $fake_db"
        fi
    done
fi

# ----------------------------
# FINAL OUTPUT
# ----------------------------

echo
echo "============================================================"
echo "Done."
echo "Campaign directory: $TESTDIR"
echo "Consolidated CSV:   $CSVFILE"
echo "============================================================"