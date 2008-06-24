#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u27_infos.log u27FJR.xml

cmsRun -j u27FJR.xml $LOCAL_TEST_DIR/u27.cfg
 
for file in u27_infos.log u27FJR.xml
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
