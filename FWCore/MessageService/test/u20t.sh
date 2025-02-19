#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u20_cerr.log FrameworkJobReport.xml

cmsRun -t -e -p $LOCAL_TEST_DIR/u20_cfg.py 2> u20_cerr.log || exit $?
 
for file in u20_cerr.log FrameworkJobReport.xml   
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
