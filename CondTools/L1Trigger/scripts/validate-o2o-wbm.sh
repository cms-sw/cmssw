#!/bin/sh

# L1Trigger O2O and WBM - validation, called by cron job

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

release=CMSSW_4_2_3_ONLINE
version=011

#cd ~zrwan/CMSSW_3_11_0/cronjob
cd /nfshome0/popcondev/L1Job/${release}/validate-o2o-wbm

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
# Last file contains last run number and its validation status
#==============================================================================

#lastFile=last.txt
lastFile=/nfshome0/popcondev/L1Job/${release}/validate-o2o-wbm/last.txt

#==============================================================================
# Summary file
#==============================================================================

#summaryFile=validate-o2o-wbm.summary
summaryFile=/nfshome0/popcondev/L1Job/o2o.summary

#==============================================================================
# Log file
#==============================================================================

#logFile=validate-o2o-wbm.log
logFile=/nfshome0/popcondev/L1Job/validate-o2o-wbm-${version}.log

#==============================================================================
# Last run number validated and its status are contained in the file last.txt.
# If the validation was failed, the contents are e.g.
# 132440 failed
# Otherwise, the contents are e.g.
# 132440 successful
# Read in the last run number validated and its status. If the status was
# successful, move forward. Otherwise, do nothing and exit, and when we
# manually check the summary file and the log file, we will find out that we
# have a problem to solve.
#==============================================================================

last=`cat ${lastFile}`

lastRun=`echo ${last} | cut -f 1 -d ' '`
lastStatus=`echo ${last} | cut -f 2 -d ' '`

#if [ $lastStatus = "failed" ]
#    then
#    exit
#fi

#==============================================================================
# In the case that last run number was validated as successful, we move
# forward, look for the first of the next run number which has not been
# validated yet.
#==============================================================================

#next=`~zrwan/CMSSW_3_11_0/cronjob/getNext.sh ${lastRun}`
next=`$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getNextO2OWBM.sh ${lastRun}`

run=`echo ${next} | cut -f 1 -d ' '`

if [ -z $run ]
    then
#    echo "`date` : validate-o2o-wbm.sh" > ${summaryFile}
#    echo "No new run to be validated" >> ${summaryFile}
    exit
fi

#==============================================================================
# Up to this point, last run number was validated as successful, and there
# is a new run number to be validated.
#==============================================================================

echo "`date` : validate-o2o-wbm.sh" >> ${logFile}
echo "run = ${run}" >> ${logFile}
echo "" >> ${logFile}

#==============================================================================
# O2O
#==============================================================================

cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py runNumber=${run} inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb_taskWriters/L1T printL1TriggerKey=1 printRSKeys=1 >& o2o.log

o2ocode=$?

cat o2o.log >> ${logFile}
echo "o2o status ${o2ocode}" >> ${logFile}
echo "" >> ${logFile}

#==============================================================================
# WBM
#==============================================================================

$CMSSW_BASE/src/CondTools/L1Trigger/scripts/wbm.sh ${run} >& wbm.log

wbmcode=$?

cat wbm.log >> ${logFile}
echo "wbm status ${wbmcode}" >> ${logFile}
echo "" >> ${logFile}

#==============================================================================
# Compare o2o.log with wbm.log, output val.log
#==============================================================================

python $CMSSW_BASE/src/CondTools/L1Trigger/scripts/validate-o2o-wbm.py

val=`cat val.log`
valStatus=`echo ${val} | cut -f 1 -d ' '`
valcode=0
if [ $valStatus = "failed" ]
    then
    valcode=1
fi

echo "val status ${valcode}" >> ${logFile}
echo "" >> ${logFile}

#==============================================================================
# Clean up and exit
#==============================================================================

rm -f o2o.log
rm -f wbm.log
rm -f val.log

exitcode=`echo ${wbmcode} + ${o2ocode} + ${valcode} | bc`

echo "run = ${run}" >> ${logFile}
echo "wbm status ${wbmcode}" >> ${logFile}
echo "o2o status ${o2ocode}" >> ${logFile}
echo "val status ${valcode}" >> ${logFile}
echo "exit code ${exitcode}" >> ${logFile}
if [ ${exitcode} -eq 0 ]
    then
    echo "L1-O2O-WBM-INFO: successful" >> ${logFile}
else
    echo "L1-O2O-WBM-INFO: failed" >> ${logFile}
fi
echo "`date` : validate-o2o-wbm.sh finished" >> ${logFile}
echo "" >> ${logFile}

tail -8 ${logFile} >> ${summaryFile}

if [ ${exitcode} -eq 0 ]
    then
    echo "${run} successful" > ${lastFile}
    # standard output goes to email
#    echo "${run} successful"
else
    echo "${run} failed" > ${lastFile}
    # standard output goes to email
#    echo "${run} failed"
fi

exit ${exitcode}
