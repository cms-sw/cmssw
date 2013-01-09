#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u11_overall_unnamed.log u11_overall_specific.log u11_supercede_specific.log u11_non_supercede_common.log u11_specific.log

cmsRun -t -p $LOCAL_TEST_DIR/u11_cfg.py || exit $?
 
for file in u11_overall_unnamed.log u11_overall_specific.log u11_supercede_specific.log u11_non_supercede_common.log u11_specific.log   
do
  sed $SED_OPT -f $LOCAL_TEST_DIR/filter-timestamps.sed $file
  diff $LOCAL_TEST_DIR/unit_test_outputs/$file $LOCAL_TMP_DIR/$file  
  if [ $? -ne 0 ]  
  then
    echo The above discrepancies concern $file 
    status=1
  fi
done

popd

exit $status
