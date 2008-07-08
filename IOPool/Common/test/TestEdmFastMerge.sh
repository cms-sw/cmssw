#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }


#---------------------------
# Create first input file
#---------------------------

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PreFastMergeTest_1_cfg.py || die 'Failure using PreFastMergeTest_1_cfg.py' $?

#---------------------------
# Create second input file
#---------------------------

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PreFastMergeTest_2_cfg.py || die 'Failure using PreFastMergeTest_2_cfg.py' $?


#---------------------------
# Merge files
#---------------------------

cmsRun -j ${LOCAL_TMP_DIR}/TestFastMergeFJR.xml --parameter-set ${LOCAL_TEST_DIR}/FastMergeTest_cfg.py || die 'Failure using FastMergeTest_cfg.py' $?
#need to filter items in job report which always change
egrep -v "<GUID>|<PFN>" $LOCAL_TEST_DIR/proper_fjr_output > $LOCAL_TMP_DIR/proper_fjr_output_filtered
egrep -v "<GUID>|<PFN>" $LOCAL_TMP_DIR/TestFastMergeFJR.xml  > $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml
diff $LOCAL_TMP_DIR/proper_fjr_output_filtered $LOCAL_TMP_DIR/TestFastMergeFJR_filtered.xml || die 'output framework job report is wrong' $?

