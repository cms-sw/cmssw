#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f u27_infos.log u27FJR.xml

cmsRun -j u27FJR.xml ${SCRAM_TEST_PATH}/u27_cfg.py || exit $?
 
for file in u27_infos.log u27FJR.xml
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
