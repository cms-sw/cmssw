#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f  u7_log.log u7_restrict.log u7_job_report.mxml

cmsRun -j u7_job_report.mxml -p $LOCAL_TEST_DIR/u7_cfg.py  || exit $?
 
for file in  u7_log.log u7_restrict.log u7_job_report.mxml
do
  sed -i -r -f $LOCAL_TEST_DIR/filter-timestamps.sed $file
  diff $LOCAL_TEST_DIR/unit_test_outputs/$file $LOCAL_TMP_DIR/$file  
  if [ $? -ne 0 ]  
  then
    echo The above discrepancies concern $file 
    status=1
  fi
done

popd

exit $status
