#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }


#---------------------------
# Create first input file
#---------------------------

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PreFastMergeTest_1_cfg.py || die 'Failure using PreFastMergeTest_1_cfg.py' $?

#---------------------------
# Create first input file with extra branch
#---------------------------

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PreFastMergeTest_1x_cfg.py || die 'Failure using PreFastMergeTest_1x_cfg.py' $?

#---------------------------
# Create second input file
#---------------------------

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PreFastMergeTest_2_cfg.py || die 'Failure using PreFastMergeTest_2_cfg.py' $?


#---------------------------
# Merge files
#---------------------------

cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJRx.xml --parameter-set ${LOCAL_TEST_DIR}/FastMergeTest_x_cfg.py || die 'Failure using FastMergeTest_x_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_fjrx_output > $LOCAL_TMP_DIR/proper_fjrx_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeFJRx.xml  > $LOCAL_TMP_DIR/TestFastMergeFJRx_filtered.xml
diff $LOCAL_TMP_DIR/proper_fjrx_output_filtered $LOCAL_TMP_DIR/TestFastMergeFJRx_filtered.xml || die 'output framework job report is wrong' $?

cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJR.xml --parameter-set ${LOCAL_TEST_DIR}/FastMergeTest_cfg.py || die 'Failure using FastMergeTest_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_fjr_output > $LOCAL_TMP_DIR/proper_fjr_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_fjr_output_filtered $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml || die 'output framework job report is wrong' $?

cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeRLFJR.xml --parameter-set ${LOCAL_TEST_DIR}/FastMergeTestRL_cfg.py || die 'Failure using FastMergeTestRL_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_RLfjr_output > $LOCAL_TMP_DIR/proper_RLfjr_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeRLFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeRLFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_RLfjr_output_filtered $LOCAL_TMP_DIR/TestFastMergeRLFJR_filtered.xml || die 'output run lumi framework job report is wrong' $?

cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeRFJR.xml --parameter-set ${LOCAL_TEST_DIR}/FastMergeTestR_cfg.py || die 'Failure using FastMergeTestR_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TEST_DIR/proper_Rfjr_output > $LOCAL_TMP_DIR/proper_Rfjr_output_filtered
egrep -v "<GUID>|<PFN>|^$" $LOCAL_TMP_DIR/TestFastMergeRFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeRFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_Rfjr_output_filtered $LOCAL_TMP_DIR/TestFastMergeRFJR_filtered.xml || die 'output run framework job report is wrong' $?
