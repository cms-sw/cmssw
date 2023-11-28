#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f u16_errors.mmlog u16_altWarnings.log u16_infos.mmlog u16_altDebugs.mmlog u16_default.log u16_job_report.mmxml u16_statistics.mslog 

cmsRun -e -j u16_job_report.mmxml ${SCRAM_TEST_PATH}/u16_cfg.py || exit $?
 
for file in u16_errors.mmlog u16_altWarnings.log u16_infos.mmlog u16_altDebugs.mmlog u16_default.log u16_job_report.mmxml u16_statistics.mslog  
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
