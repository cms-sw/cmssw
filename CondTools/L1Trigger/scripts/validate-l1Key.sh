#!/bin/sh

# L1Trigger O2O - validate L1 key

key=$1

release=CMSSW_3_5_0
emulatorRelease=CMSSW_3_5_7
version=007

logfile=/nfshome0/popcondev/L1Job/validate-l1Key-${version}.log

echo "`date` : validate-l1Key.sh $key" | tee -a ${logfile}

if [ $# -lt 1 ]
    then
    echo "Wrong number of arguments.  Usage: $0 l1key" | tee -a ${logfile}
    exit 127
fi

# set up environment variables
cd /cmsnfshome0/nfshome0/popcondev/L1Job/${release}/validate-l1Key
#export SCRAM_ARCH=slc5_ia32_gcc434
source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`

# run script; args are key tagbase records
rm -f tmpc.log

#==============================================================================
# 1. Copy conditions for a given L1 key from online database to a sqlite file
#==============================================================================

$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getConditions-l1Key.sh ${key} >& tmpc.log
o2ocode1=$?

cat tmpc.log | tee -a ${logfile}
rm -f tmpc.log

echo "getConditions-l1Key status ${o2ocode1}" | tee -a ${logfile}
echo "" | tee -a ${logfile}

#==============================================================================
# 2. Copy a raw data file from castor
#==============================================================================

# Assume Raw.root is under the current directory

#==============================================================================
# 3. Test with emulator
#==============================================================================

if [ ${o2ocode1} -eq 0 ]
    then
    cd /cmsnfshome0/nfshome0/popcondev/L1Job/${emulatorRelease}/validate-l1Key

    ln -sf /cmsnfshome0/nfshome0/popcondev/L1Job/${release}/validate-l1Key/l1config.db .
    ln -sf /cmsnfshome0/nfshome0/popcondev/L1Job/${release}/validate-l1Key/Raw.root .
    ln -sf $CMSSW_BASE/src/CondTools/L1Trigger/test/validate-l1Key.py .

    eval `scramv1 run -sh`
    cmsRun validate-l1Key.py >& tmpc.log
    o2ocode2=$?

    cat tmpc.log | tee -a ${logfile}
    rm -f tmpc.log

    echo "emulator status ${o2ocode2}" | tee -a ${logfile}
    echo "" | tee -a ${logfile}
fi

#==============================================================================
# clean up and exit
#==============================================================================

# Delete sqlite file
rm -f /cmsnfshome0/nfshome0/popcondev/L1Job/${release}/validate-l1Key/l1config.db

# Delete emulator from raw file
#rm -f /cmsnfshome0/nfshome0/popcondev/L1Job/${emulatorRelease}/validate-l1Key/L1EmulatorFromRaw.root

# Record results
echo "getConditions-l1Key status ${o2ocode1}" | tee -a ${logfile}
echo "emulator status ${o2ocode2}" | tee -a ${logfile}
o2ocode=`echo ${o2ocode1} + ${o2ocode2} | bc`
echo "exit code ${o2ocode}" | tee -a ${logfile}

if [ ${o2ocode} -eq 0 ]
    then
    echo "L1-O2O-INFO: cmsRun validate-l1Key.py successful"
else
    echo "L1-O2O-ERROR: cmsRun validate-l1Key.py failed!" >&2
fi

echo "`date` : validate-l1Key.sh finished : ${key}" | tee -a ${logfile}
echo "" | tee -a ${logfile}

tail -5 ${logfile} >> /nfshome0/popcondev/L1Job/o2o.summary

exit ${o2ocode}
