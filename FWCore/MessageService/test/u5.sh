#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u5_errors.log u5_default.log u5_noreset.log u5_reset.log 

cmsRun -p $LOCAL_TEST_DIR/u5_cfg.py || exit $?
 
for file in u5_errors.log u5_default.log u5_noreset.log u5_reset.log
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
