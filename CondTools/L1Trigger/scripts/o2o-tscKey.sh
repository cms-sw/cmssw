#!/bin/sh

# L1Trigger O2O - store TSC key

oflag=0
while getopts 'oh' OPTION
  do
  case $OPTION in
      o) oflag=1
          ;;
      h) echo "Usage: [-o] tsckey"
          echo "  -o: overwrite keys"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

# get argument
key=$1

release=CMSSW_3_5_0
version=007

echo "`date` : o2o-tscKey-slc5.sh $key" | tee -a /nfshome0/popcondev/L1Job/o2o-tscKey-${version}.log

if [ $# -lt 1 ]
    then
    echo "Wrong number of arguments.  Usage: $0 tsckey" | tee -a /nfshome0/popcondev/L1Job/o2o-tscKey-${version}.log
    exit 127
fi

# set up environment variables
cd /cmsnfshome0/nfshome0/popcondev/L1Job/${release}/o2o
export SCRAM_ARCH=slc5_ia32_gcc434
source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`

# Check for semaphore file
if [ -f o2o-tscKey.lock ]
    then
    echo "$0 already running.  Aborting process."
    exit 50
else
    touch o2o-tscKey.lock
fi

# Delete semaphore and exit if any signal is trapped
# KILL signal (9) is not trapped even though it is listed below.
trap "rm -f o2o-tscKey.lock; mv tmpb.log tmpb.log.save; exit" 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64

# run script; args are key tagbase records
rm -f tmpb.log

if [ ${oflag} -eq 0 ]
    then
    $CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-O2O-key.sh -x ${key} CRAFT09 >& tmpb.log
    o2ocode=$?
else
    $CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-O2O-key.sh -x -o ${key} CRAFT09 >& tmpb.log
    o2ocode=$?
fi

cat tmpb.log | tee -a /nfshome0/popcondev/L1Job/o2o-tscKey-${version}.log
rm -f tmpb.log

echo "cmsRun status ${o2ocode}" | tee -a /nfshome0/popcondev/L1Job/o2o-tscKey-${version}.log

if [ ${o2ocode} -eq 0 ]
    then
    echo "L1-O2O-INFO: o2o-tscKey.sh successful"
else
    echo "L1-O2O-ERROR: o2o-tscKey.sh failed!" >&2
fi

echo "`date` : o2o-tscKey-slc5.sh finished : ${key}" | tee -a /nfshome0/popcondev/L1Job/o2o-tscKey-${version}.log
echo "" | tee -a /nfshome0/popcondev/L1Job/o2o-tscKey-${version}.log

#if [ ! -f /nfshome0/popcondev/L1Job/o2o.summary ]
#    then
#    touch /nfshome0/popcondev/L1Job/o2o.summary
#fi
tail -4 /nfshome0/popcondev/L1Job/o2o-tscKey-${version}.log >> /nfshome0/popcondev/L1Job/o2o.summary

# Delete semaphore file
rm -f o2o-tscKey.lock

exit ${o2ocode}
