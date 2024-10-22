#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f  u12_warnings.log u12_placeholder.log  

cmsRun ${SCRAM_TEST_PATH}/u12_cfg.py || exit $?
 
for file in  u12_warnings.log    
do
  sed $SED_OPT -f ${SCRAM_TEST_PATH}/filter-timestamps.sed $file
  diff ${SCRAM_TEST_PATH}/unit_test_outputs/$file ./$file
  if [ $? -ne 0 ]  
  then
    echo The above discrepancies concern $file 
    status=1
  fi
done

for file in u12_placeholder.log
do
  if [ -f $file ]
  then
    echo A placeholder file was created when it should not be: $file
    status=1
  fi
done

exit $status
