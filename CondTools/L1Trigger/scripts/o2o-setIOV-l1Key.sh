#!/bin/sh

# L1Trigger O2O - set IOVs


nflag=0
oflag=""
fflag=""
xflag=""
while getopts 'nofxh' OPTION
  do
  case $OPTION in
      n) nflag=1
          ;;
      o) oflag="-o"
          ;;
      f) fflag="-f"
	  ;;
      x) xflag="-x"
	  ;;
      h) echo "Usage: [-n] runnum L1_KEY"
          echo "  -n: no RS"
          echo "  -o: overwrite RS keys"
	  echo "  -f: force IOV update"
	  echo "  -x: write to DB instead of local file"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

# arguments
run=$1
l1Key=$2

release=CMSSW_7_4_2
workdir=/nfshome0/l1emulator/run2/o2o/v7/
version=015

logfile=${workdir}/o2o-setIOV-${version}.log
summaryfile=${workdir}/o2o-summary
lockfile=o2o-setIOV.lock


echo "`date` : o2o-setIOV-l1Key.sh $run $l1Key" | tee -a ${logfile}
echo "`uptime`" | tee -a ${logfile}
START=$(date +%s)

if [ $# -lt 2 ]
    then
    echo "Wrong number of arguments.  Usage: $0 [-n] runnum L1_KEY" | tee -a ${logfile}
    exit 127
fi

# setup CMSSW
source /data/cmssw/cmsset_default.sh
cd ${workdir}/${release}
cmsenv
cd ../o2o/
SCRIPTS=${workdir}/${release}/src/CondTools/L1TriggerExt/scripts

# Check for semaphore file
if [ -f ${lockfile} ]
    then
    echo "$0 already running.  Aborting process."  | tee -a ${logfile}
    echo "$0 already running.  Aborting process."  1>&2
    tail -4 ${logfile} >> ${summaryfile}
    exit 50
else
    touch $lockfile
fi

# Delete semaphore and exit if any signal is trapped
# KILL signal (9) is not trapped even though it is listed below.
trap "rm -f ${lockfile}; mv tmp.log tmp.log.terminated; exit" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64

# run script; args are run key
rm -f tmp.log
echo "`date`" >& tmp.log

# Check if o2o-tscKey.sh is running.  If so, wait 15 seconds to prevent simultaneous writing ot ORCON.
if [ -f o2o-tscKey.lock ]
    then
    echo "o2o-tscKey.sh currently running.  Wait 15 seconds...." >> tmp.log 2>&1
    sleep 15
    echo "Resuming process." >> tmp.log 2>&1
fi


o2ocode2=0

if [ ${nflag} -eq 0 ]
    then
    echo "`date` : setting RS keys and IOVs" >> tmp.log 2>&1
    ${SCRIPTS}/runL1-O2O-rs-keysFromL1Key.sh ${xflag} ${oflag} ${fflag} ${run} ${l1Key} >> tmp.log 2>&1
    o2ocode2=$?
fi

echo "`date` : setting TSC IOVs" >> tmp.log 2>&1
tscKey=`$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getKeys.sh -t ${l1Key}`
echo "`date` : parsed tscKey = ${tscKey}" >> tmp.log 2>&1
$SCRIPTS/runL1-O2O-iov.sh ${xflag} ${oflag} ${fflag} ${run} ${tscKey} >> tmp.log 2>&1
o2ocode1=$?

tail -2 ${logfile} >> ${summaryfile}

# Filter CORAL debug output into different file, which gets deleted if no errors
grep -E "CORAL.*Info|CORAL.*Debug" tmp.log >& coraldebug-${run}.log
grep -Ev "CORAL.*Info|CORAL.*Debug" tmp.log | tee -a ${logfile}
#cat tmp.log | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log

# log TSC key and RS keys
echo "runNumber=${run} tscKey=${tscKey}" >> ./keylogs/tsckeys.txt

if [ ${nflag} -eq 0 ]
then
    grep KEYLOG tmp.log | sed 's/KEYLOG //' >> ./keylogs/rskeys.txt
fi

rm -f tmp.log

echo "cmsRun status (TSC) ${o2ocode1}" | tee -a ${logfile} 
echo "cmsRun status (RS) ${o2ocode2}" | tee -a ${logfile}
o2ocode=`echo ${o2ocode1} + ${o2ocode2} | bc`

if [ ${o2ocode} -eq 0 ]
then
    echo "L1-O2O-INFO: o2o-setIOV-l1Key-slc5.sh successful"
    rm -f coraldebug-${run}.log
else
    if [ ${o2ocode1} -eq 90 -o ${o2ocode2} -eq 90 ]
	then
	echo "L1-O2O-ERROR: problem with Oracle databases."
	echo "L1-O2O-ERROR: problem with Oracle databases." 1>&2
    else
	echo "L1-O2O-ERROR: o2o-setIOV-l1Key-slc5.sh failed!"
	echo "L1-O2O-ERROR: o2o-setIOV-l1Key-slc5.sh failed!" 1>&2
    fi
fi

echo "`date` : o2o-setIOV-l1Key-slc5.sh finished : ${run} ${l1Key}" | tee -a ${logfile}

END=$(date +%s)
DIFF=$(( $END - $START ))
if [ ${DIFF} -gt 60 ]
    then
    echo "O2O SLOW: `date`, ${DIFF} seconds for ${run} ${l1Key}" | tee -a ${logfile}
else
    echo "Time elapsed: ${DIFF} seconds" | tee -a ${logfile}
fi
echo "" | tee -a ${logfile}

tail -6 ${logfile} >> ${summaryfile}

# Delete semaphore file
rm -f ${lockfile}

exit ${o2ocode}
