#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f u1d_errors.log u1d_warnings.log u1d_infos.log u1d_debugs.log u1d_default.log u1d_job_report.mxml 

cmsRun -j u1d_job_report.mxml ${SCRAM_TEST_PATH}/u1d_cfg.py || exit $?
 
for file in u1d_errors.log u1d_warnings.log u1d_infos.log u1d_debugs.log u1d_default.log u1d_job_report.mxml   
do
  sed $SED_OPT -f ${SCRAM_TEST_PATH}/filter-timestamps.sed $file
  diff ${SCRAM_TEST_PATH}/unit_test_outputs/$file ./$file
  if [ $? -ne 0 ]  
  then
    echo The above discrepancies concern $file 
    status=1
  fi
done

exit $status
