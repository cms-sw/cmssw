#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=$SCRAM_TEST_PATH
CHANGINGPARTS="(version|microarchitecture|CPU models):"

# Need to mock the CMSSW VERSION in order to keep the process configuration ID the same
VERSION="CMSSW_20_1_0"

function run {
    CONFIG=$1
    shift
    cmsRun ${LOCAL_TEST_DIR}/${CONFIG} --version ${VERSION} $@ || die "cmsRun ${CONFIG} $@" $?
}

function compareProv {
    OPTIONS=$1
    FILE=$2
    LOG=$3
    edmRNTupleTempProvDump $OPTIONS $FILE | grep -v -E "$CHANGINGPARTS" > $LOG || die "edmProvDump $OPTIONS $FILE" $?
    diff ${LOCAL_TEST_DIR}/unit_test_outputs/$LOG $LOG  || die "comparing $LOG" $?
}

## Simple case
run testEdmProvDump_cfg.py > testEdmProvDump.log
compareProv "" testEdmProvDump.rntpl provdump_simple_default.log
compareProv --excludeESModules testEdmProvDump.rntpl provdump_simple_excludeESModules.log
compareProv --showAllModules testEdmProvDump.rntpl provdump_simple_showAllModules.log
compareProv --showTopLevelPSets testEdmProvDump.rntpl provdump_simple_showTopLevelPSets.log


## Complex case
# first processes 
run testEdmProvDump_cfg.py --ivalue 10 --accelerators=test-one --output testEdmProvDump_2.rntpl
run testEdmProvDump_cfg.py --lumi 2 --output testEdmProvDump_3.rntpl
run testEdmProvDump_cfg.py --lumi 2 --ivalue 10 --accelerators=test-two --output testEdmProvDump_4.rntpl

# first level of merge
run testEdmProvDumpMerge_cfg.py --file testEdmProvDump.rntpl --file testEdmProvDump_2.rntpl --output merged1.rntpl
run testEdmProvDumpMerge_cfg.py --ivalue 10 --file testEdmProvDump_3.rntpl --file testEdmProvDump_4.rntpl --output merged2.rntpl

compareProv "--showAllModules --showTopLevelPSets" merged1.rntpl provdump_complex_merge.log


# second level of merge
run testEdmProvDumpMerge_cfg.py --process "INTERMEDIATE" --file merged1.rntpl --file merged2.rntpl --output merged_intermediate.rntpl

compareProv "--showAllModules --showTopLevelPSets" merged_intermediate.rntpl provdump_complex_intermediate.log


# then split
run testEdmProvDumpSplit_cfg.py --process "SPLIT" --lumi 1 --file merged_intermediate.rntpl --output split1.rntpl
run testEdmProvDumpSplit_cfg.py --process "SPLIT" --lumi 2 --ivalue 9 --file merged_intermediate.rntpl --output split2.rntpl

compareProv "--showAllModules --showTopLevelPSets" split2.rntpl provdump_complex_split.log


# and merge again
run testEdmProvDumpMerge_cfg.py --process "FINAL" --file split1.rntpl --file split2.rntpl --output merged_final.rntpl

compareProv "--showAllModules --showTopLevelPSets" merged_final.rntpl provdump_complex_final.log


## One module with all parameters
run testEdmProvDumpPSet_cfg.py
compareProv "" testEdmProvDumpPSet.rntpl provdump_pset.log

exit 0
