#!/bin/bash

# Print what failed and its status
function die { echo Failure $1: status $2 ; exit $2 ; }

# 1. Generate the dependency graph
cmsRun ${SCRAM_TEST_PATH}/test_dumpDependencyGraph_cfg.py &> test_dumpDependencyGraph.log || die "cmsRun test_dumpDependencyGraph_cfg.py" $?

# 2. Do various checks on the generated dependency graph
python3 ${SCRAM_TEST_PATH}/test_dumpDependencyGraph_check.py test_dumpDependencyGraph.json || die "Check of the dependency graph JSON" $?

exit 0
