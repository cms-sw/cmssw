#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

echo "Creating files to be merged"
cmsRun ${LOCAL_TEST_DIR}/createFileForMerge_cfg.py \
    --runNumber 1 \
    --lumiBlockNumber 1 \
    --eventNumber 1 \
    --outputFile test_run1_lumi1.root || die "cmsRun createFileForMerge_cfg.py first_run1_lumi1.root" $?

cmsRun ${LOCAL_TEST_DIR}/createFileForMerge_cfg.py \
    --runNumber 1 \
    --lumiBlockNumber 2 \
    --eventNumber 1 \
    --outputFile test_run1_lumi2.root || die "cmsRun createFileForMerge_cfg.py first_run1_lumi2.root" $?

cmsRun ${LOCAL_TEST_DIR}/createFileForMerge_cfg.py \
    --runNumber 2 \
    --lumiBlockNumber 1 \
    --eventNumber 1 \
    --outputFile test_run2_lumi1.root || die "cmsRun createFileForMerge_cfg.py first_run2_lumi1.root" $?

echo "Merging files in sort order"
cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1.root test_run1_lumi2.root test_run2_lumi1.root \
    --outputFile test_merged_sort_order.root || die "cmsRun simpleMerge_cfg.py" $?

#NOTE: each transition tests three values, Thing, ThingWithMerge and ThingWithIsEqual
echo "Checking merged file in sort order"
cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_sort_order.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 102 103\
                            101 102 103\
                            101 102 103\
    --expectedEndLumiProd 1001 1002 1003\
                          1001 1002 1003\
                          1001 1002 1003\
    --verbose || die "merging (run 1, lumi 1) + (run 1, lumi 2) + (run 2, lumi 1) [i.e. sorted] failed" $?

echo "Merging files with R2 between R1 parts"
cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1.root test_run2_lumi1.root test_run1_lumi2.root \
    --outputFile test_merged_r1l1_m_r2l1_m_r1l2.root || die "cmsRun simpleMerge_cfg.py" $?

echo "Checking merged file with R2 between R1 parts"
cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_r1l1_m_r2l1_m_r1l2.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 102 103\
                            101 102 103\
                            101 102 103\
    --expectedEndLumiProd 1001 1002 1003\
                          1001 1002 1003\
                          1001 1002 1003\
    --verbose || die "merging (run 1, lumi 1) + (run 2, lumi 1) + (run 1, lumi 2) [i.e. unsorted] failed" $?

cmsRun ${LOCAL_TEST_DIR}/createFileForMerge_cfg.py \
    --runNumber 1 \
    --lumiBlockNumber 1 \
    --eventNumber 2 \
    --outputFile test_run1_lumi1_event2.root || die "cmsRun createFileForMerge_cfg.py first_run1_lumi1_event2.root" $?

echo "Merging parts lumi 1 of run 1 followed by run 2 in sort order"
cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1.root test_run1_lumi1_event2.root test_run2_lumi1.root \
    --outputFile test_merged_r1_l1_in_order.root || die "cmsRun simpleMerge_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_r1_l1_in_order.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 204 103\
                            101 102 103\
    --expectedEndLumiProd 1001 2004 1003\
                          1001 1002 1003\
    --verbose || die "merging (run 1, lumi 1) + (run 1, lumi 1) + (run 2, lumi 1) [i.e. sorted same lumi] failed" $?

echo "Merging lumi 1 of run 1 but run 2 is between the two parts of run 1"
cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1.root test_run2_lumi1.root test_run1_lumi1_event2.root \
    --outputFile test_merged_split_r1l1_m_r2l1_m_r1l1.root || die "cmsRun simpleMerge_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_split_r1l1_m_r2l1_m_r1l1.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 204 103\
                            101 102 103\
    --expectedEndLumiProd 1001 2004 1003\
                          1001 1002 1003\
    --verbose || die "merging (run 1, lumi 1) + (run 2, lumi 1) + (run 1, lumi 1) [i.e. unsorted same lumi] failed" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_split_r1l1_m_r2l1_m_r1l1.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 204 103\
                            101 102 103\
    --expectedEndLumiProd 1001 2004 1003\
                          1001 1002 1003\
    --verbose --useOutputModule || die "merging (run 1, lumi 1) + (run 2, lumi 1) + (run 1, lumi 1) [i.e. unsorted same lumi] using output module failed" $?

cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run2_lumi1.root test_run1_lumi1_event2.root \
    --outputFile test_merged_r2l1_m_r1l1.root || die "cmsRun simpleMerge_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1.root test_merged_r2l1_m_r1l1.root \
    --outputFile test_merged_split_r1l1_m_r2l1r1l1.root || die "cmsRun simpleMerge_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_split_r1l1_m_r2l1r1l1.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 204 103\
                            101 102 103\
    --expectedEndLumiProd 1001 2004 1003\
                          1001 1002 1003\
    --verbose || die "merging (run 1, lumi 1) + merged((run 2, lumi 1) + (run 1, lumi 1)) [i.e. unsorted same lumi] failed" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_split_r1l1_m_r2l1r1l1.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 204 103\
                            101 102 103\
    --expectedEndLumiProd 1001 2004 1003\
                          1001 1002 1003\
    --verbose --useOutputModule || die "merging (run 1, lumi 1) + merged((run 2, lumi 1) + (run 1, lumi 1)) [i.e. unsorted same lumi] using output module failed" $?

cmsRun ${LOCAL_TEST_DIR}/createFileForMerge_cfg.py \
    --runNumber 1 \
    --lumiBlockNumber 1 \
    --eventNumber 3 \
    --outputFile test_run1_lumi1_event3.root || die "cmsRun createFileForMerge_cfg.py first_run1_lumi1_event2.root" $?

cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1.root test_merged_r2l1_m_r1l1.root test_run1_lumi1_event3.root \
    --outputFile test_merged_r1_then_r2l1r1l1_then_r1.root || die "cmsRun simpleMerge_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_r1_then_r2l1r1l1_then_r1.root \
    --expectedBeginRunProd 10001 30006 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 300006 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 306 103\
                            101 102 103\
    --expectedEndLumiProd 1001 3006 1003\
                          1001 1002 1003\
    --verbose || die "merging (run 1,lumi 1) + merged((run 2,lumi 1) + (run 1,lumi 1)) + (run 1,lumi 1) [i.e. unsorted same lumi] failed" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_r1_then_r2l1r1l1_then_r1.root \
    --expectedBeginRunProd 10001 30006 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 300006 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 306 103\
                            101 102 103\
    --expectedEndLumiProd 1001 3006 1003\
                          1001 1002 1003\
    --verbose --useOutputModule || die "merging (run 1,lumi  1) and merge of (run 2,lumi 1, run 1,lumi 1)+ (run 1,lumi 1) using output module failed" $?

################
cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1_event2.root test_run2_lumi1.root \
    --outputFile test_merged_r1l1_m_r2l1.root || die "cmsRun simpleMerge_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/simpleMerge_cfg.py \
    --inputFiles test_run1_lumi1.root test_merged_r1l1_m_r2l1.root test_run1_lumi1_event3.root \
    --outputFile test_merged_r1_then_r1l1r2l1_then_r1.root || die "cmsRun simpleMerge_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_r1_then_r1l1r2l1_then_r1.root \
    --expectedBeginRunProd 10001 30006 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 300006 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 306 103\
                            101 102 103\
    --expectedEndLumiProd 1001 3006 1003\
                          1001 1002 1003\
    --verbose || die "merging (run 1,lumi 1) + merged((run 1,lumi 2) + (run 2,lumi 1)) + (run 1,lumi 1) [i.e. unsorted same lumi] failed" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_merged_r1_then_r1l1r2l1_then_r1.root \
    --expectedBeginRunProd 10001 30006 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 300006 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 306 103\
                            101 102 103\
    --expectedEndLumiProd 1001 3006 1003\
                          1001 1002 1003\
    --verbose --useOutputModule || die "merging (run 1,lumi 1) and merge of (run 1,lumi 1, run 2,lumi 1)+ (run 1,lumi 1) using output module failed" $?


################SPLIT between Lumi ############
#test_merged_sort_order.root

cmsRun ${LOCAL_TEST_DIR}/splitFile_cfg.py \
    --inputFile test_merged_sort_order.root \
    --outputFile test_split_r1l1_from_full_sort_order.root \
    --skipEvents 0 \
    --maxEvents 1 || die "cmsRun splitFile_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/splitFile_cfg.py \
    --inputFile test_merged_sort_order.root \
    --outputFile test_split_r1l2_from_full_sort_order.root \
    --skipEvents 1 \
    --maxEvents 1 || die "cmsRun splitFile_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/splitFile_cfg.py \
    --inputFile test_merged_sort_order.root \
    --outputFile test_split_r2l1_from_full_sort_order.root \
    --skipEvents 2 \
    --maxEvents 1 || die "cmsRun splitFile_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_split_r1l1_from_full_sort_order.root test_split_r1l2_from_full_sort_order.root test_split_r2l1_from_full_sort_order.root \
    --expectedBeginRunProd 10001 20004 10003\
                           10001 10002 10003\
    --expectedEndRunProd 100001 200004 100003\
                           100001 100002 100003\
    --expectedBeginLumiProd 101 102 103\
                            101 102 103\
                            101 102 103\
    --expectedEndLumiProd 1001 1002 1003\
                          1001 1002 1003\
                          1001 1002 1003\
    --verbose || die "splitting between Lumis failed when merging" $?

################SPLIT a Lumi ############

cmsRun ${LOCAL_TEST_DIR}/splitFile_cfg.py \
    --inputFile test_merged_r1_then_r1l1r2l1_then_r1.root \
    --outputFile test_split_r1l1e1_from_full.root \
    --skipEvents 0 \
    --maxEvents 1 || die "cmsRun splitFile_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/splitFile_cfg.py \
    --inputFile test_merged_r1_then_r1l1r2l1_then_r1.root \
    --outputFile test_split_r1l1e2_from_full.root \
    --skipEvents 1 \
    --maxEvents 1 || die "cmsRun splitFile_cfg.py" $?


cmsRun ${LOCAL_TEST_DIR}/testMergeResults_cfg.py \
    --inputFiles test_split_r1l1e1_from_full.root test_split_r1l1e2_from_full.root \
    --expectedBeginRunProd 10001 30006 10003\
    --expectedEndRunProd 100001 300006 100003\
    --expectedBeginLumiProd 101 306 103\
    --expectedEndLumiProd 1001 3006 1003\
    --verbose && die "splitting a lumi then re-merging succeeded which is unexpected (please revise this test)" 1
echo "The above failure is expected because we do not support splitting a lumi containing a mergeable Run or Lumi product"

exit 0