#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f u5_errors.log u5_default.log u5_noreset.log u5_reset.log 

cmsRun ${SCRAM_TEST_PATH}/u5_cfg.py || exit $?
 
for file in u5_errors.log u5_default.log u5_noreset.log u5_reset.log
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
