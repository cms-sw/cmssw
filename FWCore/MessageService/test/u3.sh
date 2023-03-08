#!/bin/bash

#sed on Linux and OS X have different command line options
case `uname` in Darwin) SED_OPT="-i '' -E";;*) SED_OPT="-i -r";; esac ;

status=0
  
rm -f u3_infos.log u3_statistics.log  

cmsRun -p ${SCRAM_TEST_PATH}/u3_cfg.py || exit $?
#/scratch2/mf/CMSSW_1_4_0_pre1/tmp/slc3_ia32_gcc323_dbg/src/FWCore/Framework/bin/cmsRun/cmsRun -p ${SCRAM_TEST_PATH}/u3_cfg.py || exit $?
 
for file in u3_infos.log u3_statistics.log
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
