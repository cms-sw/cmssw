#!/bin/bash

test=testSwitchProducer

function die { echo Failure $1: status $2 ; exit $2 ; }

# Running on multiple threads and streams to test any behavior that
# occurs when there are many of them
NUMTHREADS=4
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  echo "*************************************************"
  echo "SwitchProducer in a Task"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}Task_cfg.py || die "cmsRun ${test}Task_cfg.py" $?

  echo "*************************************************"
  echo "SwitchProducer in a Task, case test2 disabled"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}Task_cfg.py -- --disableTest2 || die "cmsRun ${test}Task_cfg.py -- --disableTest2" $?

  echo "*************************************************"
  echo "SwitchProducer in a Task, SwitchProducers have same labels as products in input file"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}Task_cfg.py -- --input|| die "cmsRun ${test}Task_cfg.py -- --input" $?

  echo "*************************************************"
  echo "SwitchProducer in a Task, SwitchProducers have same labels as products in input file, case test2 disabled"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}Task_cfg.py -- --input --disableTest2 || die "cmsRun ${test}TaskInput_cfg.py -- --input --disableTest2" $?

  echo "*************************************************"
  echo "Merge outputs (Task)"
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge_cfg.py outputFile=testSwitchProducerTaskMerge1.root inputFiles=file:testSwitchProducerTask1.root inputFiles=file:testSwitchProducerTask2.root || die "Merge Task1: order 1 2" $?
  echo "*************************************************"
  echo "Merge outputs in reverse order (Task)"
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge_cfg.py outputFile=testSwitchProducerTaskMerge2.root inputFiles=file:testSwitchProducerTask2.root inputFiles=file:testSwitchProducerTask1.root || die "Merge Task2: order 2 1" $?

  echo "*************************************************"
  echo "Test provenance of merged output (Task)"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerTaskMerge1.root || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py Task1" $?
  echo "*************************************************"
  echo "Test provenance of reversely merged output (Task)"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerTaskMerge2.root || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py Task2" $?


  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}Task_cfg.py -- --conditionalTask || die "cmsRun ${test}Task_cfg.py -- --conditionalTask" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, case test2 disabled"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}Task_cfg.py -- --conditionalTask --disableTest2 || die "cmsRun ${test}Task_cfg.py -- --conditionalTask --disableTest2" $?

  echo "*************************************************"
  echo "Merge outputs (ConditionalTask)"
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge_cfg.py outputFile=testSwitchProducerConditionalTaskMerge1.root inputFiles=file:testSwitchProducerConditionalTask1.root inputFiles=file:testSwitchProducerConditionalTask2.root || die "Merge ConditionalTask1: order 1 2" $?
  echo "*************************************************"
  echo "Merge outputs in reverse order (ConditionalTask)"
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge_cfg.py outputFile=testSwitchProducerConditionalTaskMerge2.root inputFiles=file:testSwitchProducerConditionalTask2.root inputFiles=file:testSwitchProducerConditionalTask1.root || die "Merge ConditionalTask2: order 2 1" $?

  echo "*************************************************"
  echo "Test provenance of merged output (ConditionalTask)"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerConditionalTaskMerge1.root || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py ConditionalTask1" $?
  echo "*************************************************"
  echo "Test provenance of reversely merged output (ConditionalTask)"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerConditionalTaskMerge2.root || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py ConditionalTask2" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, more extensive EDAlias tests"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}ConditionalTaskEDAlias_cfg.py || die "cmsRun ${test}ConditionalTaskEDAlias_cfg.py 1" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, more extensive EDAlias tests, case test2 disabled"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}ConditionalTaskEDAlias_cfg.py disableTest2 || die "cmsRun ${test}ConditionalTaskEDAlias_cfg.py 2" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, test EDAlias with all cases being explicitly consumed"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}ConditionalTaskEDAliasConsumeAllCases_cfg.py || die "cmsRun ${test}ConditionalTaskEDAliasConsumeAllCases_cfg.py 1" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, test EDAlias with all cases being explicitly consumed, case test2 disabled"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}ConditionalTaskEDAliasConsumeAllCases_cfg.py disableTest2 || die "cmsRun ${test}ConditionalTaskEDAliasConsumeAllCases_cfg.py 2" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, test EDAlias with a wildcard"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}ConditionalTaskEDAliasWildcard_cfg.py || die "cmsRun ${test}ConditionalTaskEDAliasWildcard_cfg.py" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, test EDAlias with a wildcard pattern, other module having the wildcard"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}ConditionalTaskEDAliasWildcard_cfg.py -- --patternOnOtherModule || die "cmsRun ${test}ConditionalTaskEDAliasWildcard_cfg.py --patternOnOtherModule" $?

  echo "*************************************************"
  echo "SwitchProducer in a ConditionalTask, test EDAlias with an a wildcard, case test2 disabled"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}ConditionalTaskEDAliasWildcard_cfg.py -- --disableTest2 || die "cmsRun ${test}ConditionalTaskEDAliasWildcard_cfg.py --disableTest2" $?


  
  echo "*************************************************"
  echo "SwitchProducer in a Path"
  cmsRun -n ${NUMTHREADS}  ${LOCAL_TEST_DIR}/${test}Path_cfg.py || die "cmsRun ${test}Path_cfg.py 1" $?

  echo "*************************************************"
  echo "SwitchProducer in a Path, case test2 disabled"
  cmsRun -n ${NUMTHREADS}  ${LOCAL_TEST_DIR}/${test}Path_cfg.py disableTest2 || die "cmsRun ${test}Path_cfg.py 2" $?

  echo "*************************************************"
  echo "Merge outputs (Path)"
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge_cfg.py outputFile=testSwitchProducerPathMerge1.root inputFiles=file:testSwitchProducerPath1.root inputFiles=file:testSwitchProducerPath2.root || die "Merge Path1: order 1 2" $?
  echo "*************************************************"
  echo "Merge outputs in reverse order (Path)"
  cmsRun ${LOCAL_TEST_DIR}/${test}Merge_cfg.py outputFile=testSwitchProducerPathMerge2.root inputFiles=file:testSwitchProducerPath2.root inputFiles=file:testSwitchProducerPath1.root || die "Merge Path2: order 2 1" $?

  echo "*************************************************"
  echo "Test provenance of merged output (Path)"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerPathMerge1.root || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py Path1" $?
  echo "*************************************************"
  echo "Test provenance of reversely merged output (Path)"
  cmsRun ${LOCAL_TEST_DIR}/${test}ProvenanceAnalyzer_cfg.py testSwitchProducerPathMerge2.root || die "cmsRun ${test}ProvenanceAnalyzer_cfg.py Path2" $?


  echo "*************************************************"
  echo "SwitchProducer in a Path after a failing filter"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}PathFilter_cfg.py || die "cmsRun ${test}PathFilter_cfg.py 1" $?

  echo "*************************************************"
  echo "SwitchProducer in a Path after a failing filter, case test2 disabled"
  cmsRun -n ${NUMTHREADS}  ${LOCAL_TEST_DIR}/${test}PathFilter_cfg.py disableTest2 || die "cmsRun ${test}PathFilter_cfg.py 2" $?


  echo "*************************************************"
  echo "Keeping SwitchProducer-with-EDAlias and the aliased-for product should fail"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}AliasOutput_cfg.py && die "cmsRun ${test}AliasOutput_cfg.py did not throw an exception" 1

  echo "*************************************************"
  echo "Alias to non-existent product should fail only when a corresponding product is accessed"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}AliasToNonExistent_cfg.py || die "cmsRun ${test}AliasToNonExistent_cfg.py" $?
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}AliasToNonExistent_cfg.py includeFilter && die "cmsRun ${test}AliasToNonExistent_cfg.py includeFilter did not throw an exception" 1

  echo "*************************************************"
  echo "SwitchProducer-with-EDAlias being before the aliased-for producer in a Path should fail"
  cmsRun -n ${NUMTHREADS} ${LOCAL_TEST_DIR}/${test}PathWrongOrder_cfg.py && die "cmsRun ${test}PathWrongOrder_cfg.py did not throw an exception" 1

  echo "SwitchProducer tests succeeded"
  echo "*************************************************"

exit 0
