#! /bin/bash

# Splits split_only_consumed_products_cfg.py, checks which data products each
# MPISender carries, runs the two parts over MPI, and checks that every module
# runs as many times as in the single process. The configuration describes what
# should be sent, and what should not.

set -e

HERE=$(dirname "$(realpath "$0")")
CONFIG="$HERE/split_only_consumed_products_cfg.py"
DIR=./split_only_consumed_products_result
mkdir -p $DIR
FAIL=0

REMOTE_MODULES="fromSource viaSourceAlias fromLocalOther viaLocalAlias thingsViaMixed remoteSum remoteViaAlias"
LOCAL_MODULES="checkRemoteSum checkAliasOfRemoteSum checkRemoteViaAlias"

# prints how many times the module $2 ran, from the summary in the log file $1
times_run() {
    awk '/---------- Module Summary ------------/ { f=1; next } f && $NF == "'"$2"'" { print $3; exit }' "$1"
}

# checks that the MPISenders of the configuration $1 carry exactly the data
# products $2, one "sender [(label, instance), ...]" line per MPISender; the
# MPISenders that carry no data products are not listed
check_senders() {
    local actual
    actual=$(python3 - "$1" <<'EOF'
import runpy
import sys

process = runpy.run_path(sys.argv[1])["process"]
for name, module in sorted(process.producers_().items()):
    if module.type_() == "MPISender" and len(module.products):
        print(name, sorted((p.name.getModuleLabel(), p.name.getProductInstanceLabel()) for p in module.products))
EOF
)
    echo "$actual"
    if [ "$actual" != "$2" ]; then
        echo "  FAIL: expected"
        echo "$2"
        FAIL=1
    fi
}

# 1. run the configuration in a single process
cmsRun $CONFIG > $DIR/single.log 2>&1

# 2. split the configuration, and check the data products sent in each direction
edmMpiSplitConfig $CONFIG --remote-modules $REMOTE_MODULES -l $DIR/local.py -r $DIR/remote.py
check_senders $DIR/local.py "\
mpiSenderRemote0Localsum [('localSum', 'other')]
mpiSenderRemote0Source [('source', '')]
mpiSenderRemote0Things [('things', '')]"
check_senders $DIR/remote.py "\
mpiSenderRemote0Group0 [('remoteSum', 'other'), ('remoteViaAlias', '')]"

# 3. split the configuration offloading only thingsViaMixed, and check that
#    nothing of the Source is sent
edmMpiSplitConfig $CONFIG --remote-modules thingsViaMixed -l $DIR/local_mixed.py -r $DIR/remote_mixed.py
check_senders $DIR/local_mixed.py "\
mpiSenderRemote0Things [('things', '')]"

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
check_times_run $DIR/remote_report.log $REMOTE_MODULES
check_times_run $DIR/local_report.log $LOCAL_MODULES

if [ $FAIL != 0 ]; then
    exit 1
fi
rm -rf $DIR
