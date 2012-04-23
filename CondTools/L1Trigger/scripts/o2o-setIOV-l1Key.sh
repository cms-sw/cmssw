#!/bin/sh

# L1Trigger O2O - set IOVs

nflag=0
oflag=0
pflag=0
fflag=0
while getopts 'nopfh' OPTION
  do
  case $OPTION in
      n) nflag=1
          ;;
      o) oflag=1
          ;;
      p) pflag=1
	  ;;
      f) fflag=1
	  ;;
      h) echo "Usage: [-n] runnum L1_KEY"
          echo "  -n: no RS"
          echo "  -o: overwrite RS keys"
          echo "  -p: centrally installed release, not on local machine"
	  echo "  -f: force IOV update"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

# arguments
run=$1
l1Key=$2

release=CMSSW_4_2_3_ONLINE
version=011

echo "`date` : o2o-setIOV-l1Key-slc5.sh $run $l1Key" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
START=$(date +%s)

if [ $# -lt 2 ]
    then
    echo "Wrong number of arguments.  Usage: $0 [-n] runnum L1_KEY" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
    exit 127
fi

# set up environment variables
cd /cmsnfshome0/nfshome0/popcondev/L1Job/${release}/o2o

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

# Check for semaphore file
if [ -f o2o-setIOV.lock ]
    then
    echo "$0 already running.  Aborting process."  | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
    echo "$0 already running.  Aborting process."  1>&2
    tail -3 /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log >> /nfshome0/popcondev/L1Job/o2o.summary
    exit 50
else
    touch o2o-setIOV.lock
fi

# Delete semaphore and exit if any signal is trapped
# KILL signal (9) is not trapped even though it is listed below.
trap "rm -f o2o-setIOV.lock; mv tmp.log tmp.log.save; exit" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64

# run script; args are run key
rm -f tmp.log
echo "`date` : setting TSC IOVs" >& tmp.log
tscKey=`$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getKeys.sh -t ${l1Key}`
echo "`date` : parsed tscKey = ${tscKey}" >> tmp.log 2>&1

# Check if o2o-tscKey.sh is running.  If so, wait 15 seconds to prevent simultaneous writing ot ORCON.
if [ -f o2o-tscKey.lock ]
    then
    echo "o2o-tscKey.sh currently running.  Wait 15 seconds...." >> tmp.log 2>&1
    sleep 15
    echo "Resuming process." >> tmp.log 2>&1
fi

if [ ${fflag} -eq 1 ]
    then
    forceUpdate="-f"
fi

$CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-O2O-iov.sh -x ${centralRel} ${forceUpdate} ${run} ${tscKey} >> tmp.log 2>&1
o2ocode1=$?

o2ocode2=0

if [ ${oflag} -eq 1 ]
    then
    overwrite="-o"
fi

if [ ${nflag} -eq 0 ]
    then
    echo "`date` : setting RS keys and IOVs" >> tmp.log 2>&1
    $CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-O2O-rs-keysFromL1Key.sh -x ${overwrite} ${centralRel} ${forceUpdate} ${run} ${l1Key} >> tmp.log 2>&1
    o2ocode2=$?
fi

tail -1 /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log >> /nfshome0/popcondev/L1Job/o2o.summary

# Filter CORAL debug output into different file, which gets deleted if no errors
grep -E "CORAL.*Info|CORAL.*Debug" tmp.log >& /nfshome0/popcondev/L1Job/coraldebug-${run}.log
grep -Ev "CORAL.*Info|CORAL.*Debug" tmp.log | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
#cat tmp.log | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log

# log TSC key and RS keys
echo "runNumber=${run} tscKey=${tscKey}" >> /nfshome0/popcondev/L1Job/keylogs/tsckeys.txt

if [ ${nflag} -eq 0 ]
    then
    grep KEYLOG tmp.log | sed 's/KEYLOG //' >> /nfshome0/popcondev/L1Job/keylogs/rskeys.txt
fi

rm -f tmp.log

echo "cmsRun status (TSC) ${o2ocode1}" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
echo "cmsRun status (RS) ${o2ocode2}" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
o2ocode=`echo ${o2ocode1} + ${o2ocode2} | bc`
echo "exit code ${o2ocode}" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log

if [ ${o2ocode} -eq 0 ]
    then
    echo "L1-O2O-INFO: o2o-setIOV-l1Key-slc5.sh successful"
    rm -f /nfshome0/popcondev/L1Job/coraldebug-${run}.log
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

echo "`date` : o2o-setIOV-l1Key-slc5.sh finished : ${run} ${l1Key}" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log

END=$(date +%s)
DIFF=$(( $END - $START ))
if [ ${DIFF} -gt 60 ]
    then
    echo "O2O SLOW: `date`, ${DIFF} seconds for ${run} ${l1Key}" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
else
    echo "Time elapsed: ${DIFF} seconds" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log
fi
echo "" | tee -a /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log

tail -6 /nfshome0/popcondev/L1Job/o2o-setIOV-${version}.log >> /nfshome0/popcondev/L1Job/o2o.summary

# Delete semaphore file
rm -f o2o-setIOV.lock

exit ${o2ocode}
