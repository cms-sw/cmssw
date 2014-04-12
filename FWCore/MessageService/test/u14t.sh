#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u14_errors.log u14_warnings.log u14_infos.log u14_debugs.log u14_default.log u14_job_report.mxml 

cmsRun -t -j u14_job_report.mxml -p $LOCAL_TEST_DIR/u14_cfg.py || exit $?
 
for file in u14_errors.log u14_warnings.log u14_infos.log u14_debugs.log u14_default.log u14_job_report.mxml   
do
  sed $SED_OPT -f $LOCAL_TEST_DIR/filter-timestamps.sed $file
  diff $LOCAL_TEST_DIR/unit_test_outputs/$file $LOCAL_TMP_DIR/$file  
# fancy_diffs $LOCAL_TEST_DIR/unit_test_outputs/$file $LOCAL_TMP_DIR/$file $LOCAL_TEST_DIR/$file.diffs
  if [ $? -ne 0 ]  
  then
    echo The above discrepancies concern $file 
    status=1
  fi
done

popd

exit $status
