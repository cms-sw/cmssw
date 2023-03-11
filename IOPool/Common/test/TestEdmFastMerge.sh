#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=$SCRAM_TEST_PATH
LOCAL_TMP_DIR=.

#---------------------------
# Create first input file
#---------------------------

cmsRun ${LOCAL_TEST_DIR}/PreFastMergeTest_1_cfg.py || die 'Failure using PreFastMergeTest_1_cfg.py' $?

#---------------------------
# Create first input file with extra branch
#---------------------------

cmsRun ${LOCAL_TEST_DIR}/PreFastMergeTest_1x_cfg.py || die 'Failure using PreFastMergeTest_1x_cfg.py' $?

#---------------------------
# Create second input file
#---------------------------

cmsRun ${LOCAL_TEST_DIR}/PreFastMergeTest_2_cfg.py || die 'Failure using PreFastMergeTest_2_cfg.py' $?

#---------------------------
# Create files with different ancestors
#---------------------------

cmsRun ${LOCAL_TEST_DIR}/PreFastMergeTest_ancestor1_cfg.py || die 'Failure using PreFastMergeTest_ancestor1_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/PreFastMergeTest_ancestor2_cfg.py || die 'Failure using PreFastMergeTest_ancestor2_cfg.py' $?


#---------------------------
# Merge files
#---------------------------
cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJR_ancestor.xml ${LOCAL_TEST_DIR}/FastMergeTest_ancestor_cfg.py || die 'Failure using FastMergeTest_ancestor_cfg.py' $?
cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJR_ancestor2.xml ${LOCAL_TEST_DIR}/FastMergeTest_ancestor2_cfg.py || die 'Failure using FastMergeTest_ancestor2_cfg.py' $?


cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJRx.xml ${LOCAL_TEST_DIR}/FastMergeTest_x_cfg.py || die 'Failure using FastMergeTest_x_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_fjrx_output > $LOCAL_TMP_DIR/proper_fjrx_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeFJRx.xml  > $LOCAL_TMP_DIR/TestFastMergeFJRx_filtered.xml
diff $LOCAL_TMP_DIR/proper_fjrx_output_filtered $LOCAL_TMP_DIR/TestFastMergeFJRx_filtered.xml || die 'output framework job report is wrong for proper_fjrx_output_filtered' $?

#cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJRx_second.xml ${LOCAL_TEST_DIR}/FastMergeTest_x_second_cfg.py || die 'Failure using FastMergeTest_x_second_cfg.py' $?
##need to filter items in job report which always change
#egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_fjrx_second_output > $LOCAL_TMP_DIR/proper_fjrx_second_output_filtered
#egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeFJRx_second.xml  > $LOCAL_TMP_DIR/TestFastMergeFJRx_second_filtered.xml
#diff $LOCAL_TMP_DIR/proper_fjrx_second_output_filtered $LOCAL_TMP_DIR/TestFastMergeFJRx_second_filtered.xml || die 'output framework job report is wrong for proper_fjrx_second_output_filtered' $?


cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJR.xml ${LOCAL_TEST_DIR}/FastMergeTest_cfg.py || die 'Failure using FastMergeTest_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_fjr_output > $LOCAL_TMP_DIR/proper_fjr_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_fjr_output_filtered $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml || die 'output framework job report is wrong for proper_fjr_output_filtered' $?

cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeRLFJR.xml ${LOCAL_TEST_DIR}/FastMergeTestRL_cfg.py || die 'Failure using FastMergeTestRL_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_RLfjr_output > $LOCAL_TMP_DIR/proper_RLfjr_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeRLFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeRLFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_RLfjr_output_filtered $LOCAL_TMP_DIR/TestFastMergeRLFJR_filtered.xml || die 'output run lumi framework job report is wrong for proper_RLfjr_output_filtered' $?

cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeRFJR.xml ${LOCAL_TEST_DIR}/FastMergeTestR_cfg.py || die 'Failure using FastMergeTestR_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_Rfjr_output > $LOCAL_TMP_DIR/proper_Rfjr_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeRFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeRFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_Rfjr_output_filtered $LOCAL_TMP_DIR/TestFastMergeRFJR_filtered.xml || die 'output run framework job report is wrong for proper_Rfjr_output_filtered' $?

cmsRun -p ${LOCAL_TEST_DIR}/ReadFastMergeTestOutput_cfg.py || die 'Failure using ReadFastMergeTestOutput_cfg.py' $?
