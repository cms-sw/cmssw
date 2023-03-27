#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f u14_errors.log u14_warnings.log u14_infos.log u14_debugs.log u14_default.log u14_job_report.mxml 

cmsRun -j u14_job_report.mxml -p ${SCRAM_TEST_PATH}/u14_cfg.py || exit $?
 
for file in u14_errors.log u14_warnings.log u14_infos.log u14_debugs.log u14_default.log u14_job_report.mxml   
do
  sed $SED_OPT -f ${SCRAM_TEST_PATH}/filter-timestamps.sed $file
  diff ${SCRAM_TEST_PATH}/unit_test_outputs/$file ./$file
# fancy_diffs ${SCRAM_TEST_PATH}/unit_test_outputs/$file ./$file $LOCAL_TEST_DIR/$file.diffs
  if [ $? -ne 0 ]  
  then
    echo The above discrepancies concern $file 
    status=1
  fi
done

exit $status
