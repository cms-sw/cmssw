#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f warnings.log infos.log job_report.xml 

cmsRun -j job_report.xml -p $LOCAL_TEST_DIR/u9_cfg.py
 
for file in warnings.log infos.log job_report.xml   
do
  diff $LOCAL_TEST_DIR/unit_test_outputs/$file $LOCAL_TMP_DIR/$file  
  if [ $? -ne 0 ]  
  then
    echo The above discrepancies concern $file 
    status=1
  fi
done

popd

exit $status
