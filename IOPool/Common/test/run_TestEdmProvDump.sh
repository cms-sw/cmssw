#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=$SCRAM_TEST_PATH
CHANGINGPARTS="(version|microarchitecture|CPU models):"

# Need to mock the CMSSW VERSION in order to keep the process configuration ID the same
VERSION="CMSSW_15_1_0"

function run {
    CONFIG=$1
    shift
    cmsRun ${LOCAL_TEST_DIR}/${CONFIG} --version ${VERSION} $@ || die "cmsRun ${CONFIG} $@" $?
}

function compareProv {
    OPTIONS=$1
    FILE=$2
    LOG=$3
    edmProvDump $OPTIONS $FILE | grep -v -E "$CHANGINGPARTS" > $LOG || die "edmProvDump $OPTIONS $FILE" $?
    diff ${LOCAL_TEST_DIR}/unit_test_outputs/$LOG $LOG  || die "comparing $LOG" $?
}

# Some outputs may legitimately match one of several reference files, e.g.
# because a FileInPath target resolves differently (and, in turn, some PSet
# ids that are hashed from it differ) depending on whether the source package
# providing the file is checked out in the local developer area or only
# available from the release area. compareProvAny accepts a list of candidate
# reference file names and passes if the actual output matches any of them.
function compareProvAny {
    OPTIONS=$1
    FILE=$2
    LOG=$3
    shift 3
    REFS=("$@")

    edmProvDump $OPTIONS $FILE | grep -v -E "$CHANGINGPARTS" > $LOG || die "edmProvDump $OPTIONS $FILE" $?

    for REF in "${REFS[@]}"; do
        if diff ${LOCAL_TEST_DIR}/unit_test_outputs/$REF $LOG > /dev/null 2>&1; then
            return 0
        fi
    done

    echo "Output $LOG did not match any of: ${REFS[*]}"
    echo "--- Diff against first candidate (${REFS[0]}) for inspection ---"
    diff ${LOCAL_TEST_DIR}/unit_test_outputs/${REFS[0]} $LOG
    die "comparing $LOG" 1
}

## Simple case
run testEdmProvDump_cfg.py > testEdmProvDump.log
compareProv "" testEdmProvDump.root provdump_simple_default.log
compareProv --excludeESModules testEdmProvDump.root provdump_simple_excludeESModules.log
compareProv --showAllModules testEdmProvDump.root provdump_simple_showAllModules.log
compareProv --showTopLevelPSets testEdmProvDump.root provdump_simple_showTopLevelPSets.log


## Complex case
# first processes 
run testEdmProvDump_cfg.py --ivalue 10 --accelerators=test-one --output testEdmProvDump_2.root
run testEdmProvDump_cfg.py --lumi 2 --output testEdmProvDump_3.root
run testEdmProvDump_cfg.py --lumi 2 --ivalue 10 --accelerators=test-two --output testEdmProvDump_4.root

# first level of merge
run testEdmProvDumpMerge_cfg.py --file testEdmProvDump.root --file testEdmProvDump_2.root --output merged1.root
run testEdmProvDumpMerge_cfg.py --ivalue 10 --file testEdmProvDump_3.root --file testEdmProvDump_4.root --output merged2.root

compareProv "--showAllModules --showTopLevelPSets" merged1.root provdump_complex_merge.log


# second level of merge
run testEdmProvDumpMerge_cfg.py --process "INTERMEDIATE" --file merged1.root --file merged2.root --output merged_intermediate.root

compareProv "--showAllModules --showTopLevelPSets" merged_intermediate.root provdump_complex_intermediate.log


# then split
run testEdmProvDumpSplit_cfg.py --process "SPLIT" --lumi 1 --file merged_intermediate.root --output split1.root
run testEdmProvDumpSplit_cfg.py --process "SPLIT" --lumi 2 --ivalue 9 --file merged_intermediate.root --output split2.root

compareProv "--showAllModules --showTopLevelPSets" split2.root provdump_complex_split.log


# and merge again
run testEdmProvDumpMerge_cfg.py --process "FINAL" --file split1.root --file split2.root --output merged_final.root

compareProv "--showAllModules --showTopLevelPSets" merged_final.root provdump_complex_final.log


## One module with all parameters
run testEdmProvDumpPSet_cfg.py
compareProvAny "" testEdmProvDumpPSet.root provdump_pset.log \
    provdump_pset_fiplocal.log provdump_pset_fiprelease.log

exit 0
