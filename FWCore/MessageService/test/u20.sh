#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
if [ "$UBSAN_OPTIONS" != "" ] ; then export UBSAN_OPTIONS="log_path=ubsan.log:${UBSAN_OPTIONS}"; fi
rm -f u20_cerr.log FrameworkJobReport.xml

cmsRun -e ${SCRAM_TEST_PATH}/u20_cfg.py 2> u20_cerr.log || exit $?
sed -n '/Disabling gnu++: clang has no __float128 support on this target!/!p' u20_cerr.log > tmpf && mv tmpf u20_cerr.log # remove clang output

for file in u20_cerr.log FrameworkJobReport.xml   
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
