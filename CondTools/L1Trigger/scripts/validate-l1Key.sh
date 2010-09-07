#!/bin/sh

# L1Trigger O2O - validate L1 key, called by cron job

#==============================================================================
# Environment
#==============================================================================

release=CMSSW_3_5_0
emulatorRelease=CMSSW_3_5_7
version=007

#==============================================================================
# Last file contains last time stamp, its key, and its validation status
#==============================================================================

#lastFile=~zrwan/CMSSW_3_5_0/cronjob/last.txt
lastFile=~popcondev/L1Job/${release}/validate-l1Key/last.txt

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
# Last creation date validated, its L1 key, and its status are contained in the
# file last.txt. If the validation was failed, the contents are e.g.
# 2010.08.19_16:28:59_730020000 TSC_20100818_002298_cosmics_BASE failed
# Otherwise, the contents are e.g.
# 2010.08.19_16:28:59_730020000 TSC_20100818_002298_cosmics_BASE successful
# Read in the last creation date validated, its L1 key, and its status. If the
# status was successful, assume all of the L1 keys for the previous creation
# dates were validated successfully already and move forward. Otherwise, do
# nothing and exit, and when we manually check the summary file and the log
# file, we will find out that we have a problem to solve.
#==============================================================================

last=`cat ${lastFile}`

lastCreationDate=`echo ${last} | cut -f 1 -d ' '`
lastTscKey=`echo ${last} | cut -f 2 -d ' '`
lastStatus=`echo ${last} | cut -f 3 -d ' '`

if [ $lastStatus == "failed" ]
    then
    echo "Last validated key failed: ${lastCreationDate} ${lastTscKey}" > ${summaryFile}
    exit
fi

#==============================================================================
# In the case that last creation date was validated as successful, we move
# forward, look for the first of the next new L1 keys which has not been
# validated yet.
#==============================================================================

#next=`~zrwan/CMSSW_3_5_0/cronjob/getNext.sh ${lastCreationDate}`
next=`$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getNextTscKeyByTimestamp.sh ${lastCreationDate}`

creation_date=`echo $next | cut -f 1 -d ' '`
tsc_key=`echo $next | cut -f 2 -d ' '`

if [ -z $creation_date ]
    then
#    echo "`date` : validate-l1Key.sh" > ${summaryFile}
#    echo "No new key to be validated" >> ${summaryFile}
    exit
fi

#==============================================================================
# Up to this point, last creation date was validated as successful, and there
# is a new creation date with a new L1 key to be validated.
#==============================================================================

echo "`date` : validate-l1Key.sh" >> ${logFile}
echo "creation_date = ${creation_date}" >> ${logFile}
echo "tsc_key = ${tsc_key}" >> ${logFile}

#==============================================================================
# Set up environment
#==============================================================================

#cd ~zrwan/CMSSW_3_5_0/cronjob
cd /nfshome0/popcondev/L1Job/${release}/validate-l1Key
source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`

#==============================================================================
# 1. Copy conditions for a given TSC key from online database to a sqlite file
#==============================================================================

rm -f temp.log

$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getConditions.sh -n ${tsc_key} >& temp.log
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
    cd /nfshome0/popcondev/L1Job/${emulatorRelease}/validate-l1Key
    ln -sf /nfshome0/popcondev/L1Job/${release}/validate-l1Key/l1config.db .
    ln -sf /nfshome0/popcondev/L1Job/${release}/validate-l1Key/Raw.root .
    ln -sf $CMSSW_BASE/src/CondTools/L1Trigger/test/validate-l1Key.py .

    eval `scramv1 run -sh`
    cmsRun validate-l1Key.py >& temp.log
    o2ocode2=$?

    cat temp.log >> ${logFile}
    rm -f temp.log

    echo "emulator status ${o2ocode2}" >> ${logFile}
    echo "" >> ${logFile}
fi

#==============================================================================
# Clean up and exit
#==============================================================================

#rm -f ~zrwan/CMSSW_3_5_0/cronjob/l1config.db
rm -f /nfshome0/popcondev/L1Job/${release}/validate-l1Key/l1config.db

o2ocode=`echo ${o2ocode1} + ${o2ocode2} | bc`

echo "creation_date = ${creation_date}" >> ${logFile}
echo "tsc_key = ${tsc_key}" >> ${logFile}
echo "getConditions status ${o2ocode1}" >> ${logFile}
echo "emulator status ${o2ocode2}" >> ${logFile}
echo "exit code ${o2ocode}" >> ${logFile}
if [ ${o2ocode} -eq 0 ]
    then
    echo "L1-O2O-INFO: successful" >> ${logFile}
else
    echo "L1-O2O-INFO: failed" >> ${logFile}
fi
echo "`date` : validate-l1Key.sh finished" >> ${logFile}

tail -7 ${logFile} >> ${summaryFile}

cat ${lastFile} >> ${lastFile}.done
if [ ${o2ocode} -eq 0 ]
    then
    echo "${creation_date} ${tsc_key} successful" > ${lastFile}
    # standard output goes to email
    #echo "${creation_date} ${tsc_key} successful"
else
    echo "${creation_date} ${tsc_key} failed" > ${lastFile}
    # standard output goes to email
    #echo "${creation_date} ${tsc_key} failed"
fi

exit ${o2ocode}
