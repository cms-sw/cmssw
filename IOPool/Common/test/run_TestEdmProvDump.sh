#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=$SCRAM_TEST_PATH
CHANGINGPARTS="(version|microarchitecture|CPU models):"

function compareProv {
    OPTIONS=$1
    FILE=$2
    LOG=$3
    edmProvDump $OPTIONS $FILE | grep -v -E "$CHANGINGPARTS" > $LOG || die "edmProvDump $OPTIONS $FILE" $?
    diff ${LOCAL_TEST_DIR}/unit_test_outputs/$LOG $LOG  || die "comparing $LOG" $?
}

## Simple case
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDump_cfg.py > testEdmProvDump.log || die "cmsRun testEdmProvDump_cfg.py" $?
compareProv "" testEdmProvDump.root provdump_simple_default.log
compareProv --excludeESModules testEdmProvDump.root provdump_simple_excludeESModules.log
compareProv --showAllModules testEdmProvDump.root provdump_simple_showAllModules.log
compareProv --showTopLevelPSets testEdmProvDump.root provdump_simple_showTopLevelPSets.log


## Complex case
# first processes 
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDump_cfg.py --ivalue 10 --accelerators=test-one --output testEdmProvDump_2.root || die "cmsRun testEdmProvDump_cfg.py --ivalue 10" $?
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDump_cfg.py --lumi 2 --output testEdmProvDump_3.root || die "cmsRun testEdmProvDump_cfg.py --lumi 2" $?
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDump_cfg.py --lumi 2 --ivalue 10 --accelerators=test-two --output testEdmProvDump_4.root || die "cmsRun testEdmProvDump_cfg.py --lumi 2 --ivalue 10" $?

# first level of merge
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDumpMerge_cfg.py --file testEdmProvDump.root --file testEdmProvDump_2.root --output merged1.root || die "cmsRun testEdmProvDumpMerge_cfg.py --file testEdmProvDump.root --file testEdmProvDump_2.root --output merged1.root" $?
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDumpMerge_cfg.py --ivalue 10 --file testEdmProvDump_3.root --file testEdmProvDump_4.root --output merged2.root || die "cmsRun testEdmProvDumpMerge_cfg.py --ivalue 10 --file testEdmProvDump_3.root --file testEdmProvDump_4.root --output merged2.root" $?

compareProv "--showAllModules --showTopLevelPSets" merged1.root provdump_complex_merge.log


# second level of merge
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDumpMerge_cfg.py --process "INTERMEDIATE" --file merged1.root --file merged2.root --output merged_intermediate.root || die "cmsRun testEdmProvDumpMerge_cfg.py --file merged1.root --file merged2.root --output merged_intermediate.root" $?

compareProv "--showAllModules --showTopLevelPSets" merged_intermediate.root provdump_complex_intermediate.log


# then split
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDumpSplit_cfg.py --process "SPLIT" --lumi 1 --file merged_intermediate.root --output split1.root || die "cmsRun testEdmProvDumpSplit_cfg.py --lumi 1 --output split1.root" $?
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDumpSplit_cfg.py --process "SPLIT" --lumi 2 --ivalue 9 --file merged_intermediate.root --output split2.root || die "cmsRun testEdmProvDumpSplit_cfg.py --lumi 2 --ivalue 9 --output split2.root" $?

compareProv "--showAllModules --showTopLevelPSets" split2.root provdump_complex_split.log


# and merge again
cmsRun ${LOCAL_TEST_DIR}/testEdmProvDumpMerge_cfg.py --process "FINAL" --file split1.root --file split2.root --output merged_final.root || die "cmsRun testEdmProvDumpMerge_cfg.py --file split1.root --file split2.root --output merged_final.root" $?

compareProv "--showAllModules --showTopLevelPSets" merged_final.root provdump_complex_final.log

exit 0
