#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

pushd $LOCAL_TMP_DIR

status=0
  
rm -f u20_cerr.log FrameworkJobReport.xml

cmsRun -e -p $LOCAL_TEST_DIR/u20_cfg.py 2> u20_cerr.log || exit $?
sed -n '/Disabling gnu++: clang has no __float128 support on this target!/!p' u20_cerr.log > tmpf && mv tmpf u20_cerr.log # remove clang output

for file in u20_cerr.log FrameworkJobReport.xml   
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
