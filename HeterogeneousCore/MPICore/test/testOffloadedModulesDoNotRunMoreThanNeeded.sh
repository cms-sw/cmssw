#! /bin/bash

# Splits offloaded_modules_do_not_run_more_than_needed_cfg.py, runs the two parts
# over MPI, and checks that every module runs as many times as in the single
# process. The configuration describes what should happen.

set -e

HERE=$(dirname "$(realpath "$0")")
CONFIG="$HERE/offloaded_modules_do_not_run_more_than_needed_cfg.py"
DIR=./offloaded_modules_do_not_run_more_than_needed_result
mkdir -p $DIR
FAIL=0

# prints how many times the module $2 ran, from the summary in the log file $1
times_run() {
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

# 1. run the configuration in a single process
cmsRun $CONFIG > $DIR/single.log 2>&1

# 2. split the configuration
edmMpiSplitConfig $CONFIG --remote-modules producerB producerC producerD producerE \
    -l $DIR/local.py -r $DIR/remote.py

# 3. check that the offloaded modules form two groups: each group has an
#    activity capture and a sender in the local process, and a receiver and an
#    activity filter in the remote one
check_two_groups() {
    local count
    count=$(grep -c "process\.$2 = cms\." "$1" || true)
    echo "$2: $count (expected 2)"
    if [ "$count" != 2 ]; then
        FAIL=1
    fi
}
check_two_groups $DIR/local.py "activityCaptureBeforeRemote0Group[0-9]*"
check_two_groups $DIR/local.py "mpiSenderRemote0Group[0-9]*Activity"
check_two_groups $DIR/remote.py "mpiReceiverRemote0Group[0-9]*Activity"
check_two_groups $DIR/remote.py "activityFilterBeforeRemote0Group[0-9]*"

# 4. run the two parts over MPI, each one writing its summary to its own log file
echo 'process.MessageLogger.files.local_report = cms.untracked.PSet()' >> $DIR/local.py
echo 'process.MessageLogger.files.remote_report = cms.untracked.PSet()' >> $DIR/remote.py
if ! (cd $DIR && "$HERE/testMPICommWorld.sh" local.py remote.py) > $DIR/mpi_run.log 2>&1; then
    echo "FAIL: the split processes did not run to completion; see $DIR/mpi_run.log"
    exit 1
fi

# 5. check that every module ran as many times as in the single process
check_times_run() {
    local log=$1 module expected actual
    shift
    for module in "$@"; do
        expected=$(times_run $DIR/single.log $module)
        actual=$(times_run $log $module)
        echo "$module ran $actual times (expected $expected)"
        if [ -z "$actual" ] || [ "$actual" != "$expected" ]; then
            echo "  FAIL: mismatch for $module"
            FAIL=1
        fi
    done
}
check_times_run $DIR/remote_report.log producerB producerC producerD producerE
check_times_run $DIR/local_report.log producerF

if [ $FAIL != 0 ]; then
    exit 1
fi
rm -rf $DIR
