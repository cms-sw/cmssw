#!/bin/sh

# L1Trigger O2O - validate TSC key, and write it to ORCON called by cron job on l1-o2o.cms

# Run as l1emulator on machine with frontier squids enabled.  Check with `rpm -qa | grep squid`.
# Should have a line with STABLE, e.g. squid-2.7.STABLE7-frontiercmshlt2

pflag=0
while getopts 'ph' OPTION
  do
  case $OPTION in
      p) pflag=1
          ;;
      h) echo "Usage: [-p]"
          echo "  -p: centrally installed release, not on local machine"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

#==============================================================================
# Environment
#==============================================================================

release=CMSSW_3_8_1_onlpatch4_ONLINE
emulatorRelease=CMSSW_3_8_2
version=009

#==============================================================================
# File containing TSC keys that have been tested and written (+ status)
#==============================================================================

# writtenFile=~zrwan/CMSSW_3_5_0/cronjob/writtenTscKeys.txt
writtenFile=~popcondev/L1Job/${release}/validate-l1Key/writtenTscKeys.txt

#==============================================================================
# Summary file
#==============================================================================

#summaryFile=~zrwan/CMSSW_3_5_0/cronjob/o2o.summary
summaryFile=/nfshome0/popcondev/L1Job/o2o.summary

#==============================================================================
# Log file
#==============================================================================

#logFile=~zrwan/CMSSW_3_5_0/cronjob/validate-l1Key-${version}.log
logFile=/nfshome0/popcondev/L1Job/validate-l1Key-${version}.log

#==============================================================================
# Set up environment
#==============================================================================

#cd ~zrwan/CMSSW_3_5_0/cronjob
cd /nfshome0/popcondev/L1Job/${release}/validate-l1Key

if [ ${pflag} -eq 0 ]
    then
    export SCRAM_ARCH=""
    export VO_CMS_SW_DIR=""
    source /opt/cmssw/cmsset_default.sh
else
    source /nfshome0/cmssw2/scripts/setup.sh
    centralRel="-p"
fi
eval `scramv1 run -sh`

#==============================================================================
# Find next key to be tested and written
#==============================================================================

if [ ! -f ${writtenFile} ]
    then
    touch ${writtenFile}
fi

#for KEY in `~zrwan/CMSSW_3_5_0/cronjob/getValidTscKeys.sh`
for KEY in `$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getValidTscKeys.sh`
do
  if [ -z "${tsc_key}" ]
      then
      last=`grep ${KEY} writtenTscKeys.txt`
      if [ -n "${last}" ]
	  then
	  status=`echo ${last} | cut -f 2 -d ' '`
          #  echo "Key ${KEY} already written and validated: status ${status}"
      else
          #  echo "Setting tsc_key ${KEY}"
	  tsc_key=${KEY}
      fi
  fi
done

if [ -z "${tsc_key}" ]
    then
    # echo "No new keys to write and test." >> ${summaryFile}
    exit 0
fi

#==============================================================================
# Start testing and writing new key
#==============================================================================

echo "`date` : validate-l1Key.sh" >> ${logFile}
echo "tsc_key = ${tsc_key}" >> ${logFile}
echo "O2O release ${release}$" >> ${logFile}

#==============================================================================
# 1. Copy conditions for a given TSC key from online database to a sqlite file
#==============================================================================

rm -f temp.log

$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getConditions.sh -n ${centralRel} ${tsc_key} >& temp.log
o2ocode1=$?

cat temp.log >> ${logFile}
rm -f temp.log

echo "getConditions status ${o2ocode1}" >> ${logFile}
echo "" >> ${logFile}

#==============================================================================
# 2. Copy a raw data file from castor
#==============================================================================

# Assume Raw.root is under the current directory

#==============================================================================
# 3. Test with emulator
#==============================================================================

if [ ${o2ocode1} -eq 0 ]
    then
    echo "Running emulator job in ${emulatorRelease}" >> ${logFile}
    cd /nfshome0/popcondev/L1Job/${emulatorRelease}/validate-l1Key
    ln -sf /nfshome0/popcondev/L1Job/${release}/validate-l1Key/l1config.db .
    ln -sf /nfshome0/popcondev/L1Job/${release}/validate-l1Key/Raw.root .
    ln -sf $CMSSW_BASE/src/CondTools/L1Trigger/test/validate-l1Key.py .

    export SCRAM_ARCH=""
    export VO_CMS_SW_DIR=""
    source /nfshome0/cmssw/scripts/setup.sh

    eval `scramv1 run -sh`
    cmsRun validate-l1Key.py >& temp.log
    o2ocode2=$?

    cat temp.log >> ${logFile}
    rm -f temp.log

    echo "emulator status ${o2ocode2}" >> ${logFile}
    echo "" >> ${logFile}
fi

#==============================================================================
# 4. Clean up
#==============================================================================

#rm -f ~zrwan/CMSSW_3_5_0/cronjob/l1config.db
rm -f /nfshome0/popcondev/L1Job/${release}/validate-l1Key/l1config.db

o2ocode=`echo ${o2ocode1} + ${o2ocode2} | bc`

echo "tsc_key = ${tsc_key}" >> ${logFile}
echo "getConditions status ${o2ocode1}" >> ${logFile}
echo "emulator status ${o2ocode2}" >> ${logFile}
echo "exit code ${o2ocode}" >> ${logFile}
if [ ${o2ocode} -eq 0 ]
    then
    echo "L1-O2O-INFO: successful" >> ${logFile}
    validationStatus="successful"
else
    echo "L1-O2O-INFO: failed" >> ${logFile}
    validationStatus="failed"
    writeStatus="failed"
fi
echo "`date` : validate-l1Key.sh finished" >> ${logFile}

tail -6 ${logFile} >> ${summaryFile}
echo "" >> ${logFile}

#==============================================================================
# 5. If key is validated, call O2O payload-writing script
#==============================================================================

if [ ${o2ocode} -eq 0 ]
    then
    /nfshome0/l1emulator/o2o/o2o-tscKey-slc5.sh ${tsc_key}
    o2ocode3=$?

    if [ ${o2ocode3} -eq 0 ]
	then
	writeStatus="successful"
    else
	writeStatus="failed"
    fi

    o2ocode=`echo ${o2ocode} + ${o2ocode3} | bc`
fi

echo "${tsc_key} ${validationStatus} ${writeStatus}" >> ${writtenFile}

exit ${o2ocode}
