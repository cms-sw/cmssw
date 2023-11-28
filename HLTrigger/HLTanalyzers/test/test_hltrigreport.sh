#!/bin/sh

function die { echo $1: status $2; exit $2; }
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
cp ${LOCAL_TEST_DIR}/test_hltrigreport_base_cfg.py .

echo reset:event report:event
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_event_event.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/event_event.log || die 'failed running cmsRun test_hltrigreport_event_event.py' $?

echo reset:event report:lumi
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_event_lumi.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/event_lumi.log || die 'failed running cmsRun test_hltrigreport_event_lumi.py' $?

echo reset:event report:run
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_event_run.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/event_run.log || die 'failed running cmsRun test_hltrigreport_event_run.py' $?

echo reset:event report:job
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_event_job.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/event_job.log || die 'failed running cmsRun test_hltrigreport_event_job.py' $?

echo reset:lumi report:event
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_lumi_event.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/lumi_event.log || die 'failed running cmsRun test_hltrigreport_lumi_event.py' $?

echo reset:lumi report:lumi
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_lumi_lumi.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/lumi_lumi.log || die 'failed running cmsRun test_hltrigreport_lumi_lumi.py' $?

echo reset:lumi report:run
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_lumi_run.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/lumi_run.log || die 'failed running cmsRun test_hltrigreport_lumi_run.py' $?

echo reset:lumi report:job
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_lumi_job.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/lumi_job.log || die 'failed running cmsRun test_hltrigreport_lumi_job.py' $?

echo reset:run report:event
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_run_event.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/run_event.log || die 'failed running cmsRun test_hltrigreport_run_event.py' $?

echo reset:run report:lumi
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_run_lumi.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/run_lumi.log || die 'failed running cmsRun test_hltrigreport_run_lumi.py' $?

echo reset:run report:run
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_run_run.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/run_run.log || die 'failed running cmsRun test_hltrigreport_run_run.py' $?

echo reset:run report:job
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_run_job.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/run_job.log || die 'failed running cmsRun test_hltrigreport_run_job.py' $?

echo reset:never report:event
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_never_event.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/never_event.log || die 'failed running cmsRun test_hltrigreport_never_event.py' $?

echo reset:never report:lumi
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_never_lumi.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/never_lumi.log || die 'failed running cmsRun test_hltrigreport_never_lumi.py' $?

echo reset:never report:run
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_never_run.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/never_run.log || die 'failed running cmsRun test_hltrigreport_never_run.py' $?


echo reset:never report:job
(cmsRun ${LOCAL_TEST_DIR}/test_hltrigreport_never_job.py 2>&1) | grep 'HLT-Report' | diff - ${LOCAL_TEST_DIR}/comparison_logs/never_job.log || die 'failed running cmsRun test_hltrigreport_never_job.py' $?
