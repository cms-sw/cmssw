#!/bin/bash

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u3_infos.log u3_statistics.log  

cmsRun -p $LOCAL_TEST_DIR/u3_cfg.py || exit $?
#/scratch2/mf/CMSSW_1_4_0_pre1/tmp/slc3_ia32_gcc323_dbg/src/FWCore/Framework/bin/cmsRun/cmsRun -p $LOCAL_TEST_DIR/u3_cfg.py || exit $?
 
for file in u3_infos.log u3_statistics.log
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
