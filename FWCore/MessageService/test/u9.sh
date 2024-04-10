#!/bin/bash

case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f warnings.log infos.log job_report.xml 

cmsRun -j job_report.xml ${SCRAM_TEST_PATH}/u9_cfg.py || exit $?
 
for file in warnings.log infos.log job_report.xml   
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
