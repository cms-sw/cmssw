#!/bin/bash

test=testEDAlias

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo "*************************************************"
  echo "EDAlias consumer in a Task"
  cmsRun ${LOCAL_TEST_DIR}/${test}Task_cfg.py || die "cmsRun ${test}Task_cfg.py 1" $?

  echo "*************************************************"
  echo "Test output"
  cmsRun ${LOCAL_TEST_DIR}/${test}Analyze_cfg.py testEDAliasTask.root || die "cmsRun ${test}Analyze_cfg.py testEDAliasTask.root" $?

  echo "*************************************************"
  echo "EDAlias consumer in a Path"
  cmsRun ${LOCAL_TEST_DIR}/${test}Path_cfg.py || die "cmsRun ${test}Path_cfg.py 1" $?

  echo "*************************************************"
  echo "Test output"
  cmsRun ${LOCAL_TEST_DIR}/${test}Analyze_cfg.py testEDAliasTask.root || die "cmsRun ${test}Analyze_cfg.py testEDAliasPath.root" $?

popd

exit 0
