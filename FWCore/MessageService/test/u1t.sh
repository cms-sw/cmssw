#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u1_errors.log u1_warnings.log u1_infos.log u1_debugs.log u1_default.log u1_job_report.mxml 

cmsRun -t -j u1_job_report.mxml -p $LOCAL_TEST_DIR/u1_cfg.py
 
for file in u1_errors.log u1_warnings.log u1_infos.log u1_debugs.log u1_default.log u1_job_report.mxml   
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
