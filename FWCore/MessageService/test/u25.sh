#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u25_only.log 

cmsRun -p $LOCAL_TEST_DIR/u25_cfg.py
 
for file in u25_only.log   
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
