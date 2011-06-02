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

release=CMSSW_3_11_0_ONLINE
# Emulator cannot run in online releases because of missing packages
#emulatorRelease=CMSSW_3_11_0
emulatorRelease=CMSSW_4_2_3
version=010

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
# Check for semaphore file
#==============================================================================

semaphoreFile=/nfshome0/popcondev/L1Job/${release}/validate-l1Key/validate-l1Key.lock

if [ -f ${semaphoreFile} ]
    then
    echo "`date` : validate-l1Key.sh" >> ${logFile}
    echo "$0 already running.  Aborting process.  Check for hung jobs from previous call of validate-l1Key.sh."  | tee -a ${logFile}
    tail -3 ${logFile} >> /nfshome0/popcondev/L1Job/o2o.summary
    exit 50
else
    touch ${semaphoreFile}
fi

# Delete semaphore and exit if any signal is trapped
# KILL signal (9) is not trapped even though it is listed below.
trap "rm -f ${semaphoreFile}; mv /nfshome0/popcondev/L1Job/${release}/validate-l1Key/temp.log /nfshome0/popcondev/L1Job/${release}/validate-l1Key/temp.log.save; mv /nfshome0/popcondev/L1Job/${emulatorRelease}/validate-l1Key/temp.log /nfshome0/popcondev/L1Job/${emulatorRelease}/validate-l1Key/temp.log.save; exit" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64

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
    rm -f ${semaphoreFile}
    exit 0
fi

#==============================================================================
# Start testing and writing new key
#==============================================================================

echo "`date` : validate-l1Key.sh" >> ${logFile}
echo "tsc_key = ${tsc_key}" >> ${logFile}
echo "O2O release ${release}$" >> ${logFile}

#==============================================================================
# 0. Write TSC payloads.  Decoupled from validation (for now) b/c of timing.
#==============================================================================

/nfshome0/l1emulator/o2o/o2o-tscKey-slc5.sh ${tsc_key}
o2ocode3=$?

if [ ${o2ocode3} -eq 0 ]
    then
    writeStatus="successful"
else
    writeStatus="failed"
fi

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

    export SCRAM_ARCH=slc5_amd64_gcc434
    #export VO_CMS_SW_DIR=""
    source /nfshome0/cmssw2/scripts/setup.sh

    eval `scramv1 run -sh`
    ln -sf $CMSSW_BASE/src/CondTools/L1Trigger/test/validate-l1Key.py .
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
#    writeStatus="failed"
fi
echo "`date` : validate-l1Key.sh finished" >> ${logFile}

tail -6 ${logFile} >> ${summaryFile}
echo "" >> ${logFile}

##==============================================================================
## 5. If key is validated, call O2O payload-writing script
##==============================================================================
#
#if [ ${o2ocode} -eq 0 ]
#    then
#    /nfshome0/l1emulator/o2o/o2o-tscKey-slc5.sh ${tsc_key}
#    o2ocode3=$?
#
#    if [ ${o2ocode3} -eq 0 ]
#	then
#	writeStatus="successful"
#    else
#	writeStatus="failed"
#    fi

    o2ocode=`echo ${o2ocode} + ${o2ocode3} | bc`
#fi

echo "${tsc_key} ${validationStatus} ${writeStatus}" >> ${writtenFile}

# Delete semaphore file
rm -f ${semaphoreFile}

exit ${o2ocode}
