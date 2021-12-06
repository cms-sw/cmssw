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

  echo "*************************************************"
  echo "Test EDAlias aliasing for many modules"
  cmsRun ${LOCAL_TEST_DIR}/${test}ManyModules_cfg.py || die "cmsRun ${test}ManyModules_cfg.py 1" $?

  echo "*************************************************"
  echo "Test EDAlias aliasing for many modules with possibly ambiguous get via edm::View"
  cmsRun ${LOCAL_TEST_DIR}/${test}ManyModulesAmbiguous_cfg.py && die "cmsRun ${test}ManyModulesAmbiguous_cfg.py 1" 1
  cmsRun ${LOCAL_TEST_DIR}/${test}ManyModulesAmbiguous_cfg.py includeAliasToFoo=0 || die "cmsRun ${test}ManyModulesAmbiguous_cfg.py includeAliasToFoo=0" $?
  cmsRun ${LOCAL_TEST_DIR}/${test}ManyModulesAmbiguous_cfg.py includeAliasToBar=0 || die "cmsRun ${test}ManyModulesAmbiguous_cfg.py includeAliasToBar=0" $?
  cmsRun ${LOCAL_TEST_DIR}/${test}ManyModulesAmbiguous_cfg.py consumerGets=0 || die "cmsRun ${test}ManyModulesAmbiguous_cfg.py consumerGets=0" $?
  cmsRun ${LOCAL_TEST_DIR}/${test}ManyModulesAmbiguous_cfg.py explicitProcessName=1 && die "cmsRun ${test}ManyModulesAmbiguous_cfg.py explicitProcessName=1" 1

popd

exit 0
