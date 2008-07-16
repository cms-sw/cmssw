#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u21_warnings.log u21_infos.log  

cmsRun -p $LOCAL_TEST_DIR/u21_cfg.py
 
for file in u21_warnings.log u21_infos.log   
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
