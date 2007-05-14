#!/bin/sh
####################################
#//////////////////////////////////#
####################################
##           LOCAL PATHS          ##
## change this for your local dir ##
####################################

## Where to find all the templates and to write all the logs
export LOCALHOME=/analysis/sw/CRAB
## Where to copy all the results
export MainStoreDir=/data1/CrabAnalysis
## Where to create crab jobs
export WorkingDir=/tmp/${USER}
## Leave python path as it is to source in standard (local) area
export python_path=/analysis/sw/CRAB

####################################
#//////////////////////////////////#
####################################

########################################
## Patch to make it work with crontab ##
###########################################
export MYHOME=/analysis/sw/CRAB
# the scritps will source ${MYHOME}/crab.sh
###########################################

[ -e ${LOCALHOME}/lock_summ ] && exit

touch ${LOCALHOME}/lock_summ

# Process all the Flags
for Version in `cat ${LOCALHOME}/Summary_cron.cfg | grep -v "#"`
  do

  echo -e "\n${LOCALHOME}/MakePlots.sh ${Version} > ${MainStoreDir}/logs/${Version}/LogMonitor/plots_log_`date +\%Y-\%m-\%d_\%H-\%M-\%S`"
  ${LOCALHOME}/MakePlots.sh ${Version} > ${MainStoreDir}/logs/${Version}/LogMonitor/plots_log_`date +\%Y-\%m-\%d_\%H-\%M-\%S`

  echo -e "\n...Running BadStripsFromPosition $Version"
  ${LOCALHOME}/macros/BadStripsFromPosition.sh ${Version}
  
  echo -e "\n...Running BadStripsFromDBNoise $Version"
  ${LOCALHOME}/macros/BadStripsFromDBNoise.sh ${Version}

  echo -e "\n...Running BadModulesFromPedestals $Version"
  ${LOCALHOME}/macros/BadModulesFromPedestals.sh ${Version}

  echo -e "\n...Running BadModulesFromClusters $Version"
  ${LOCALHOME}/macros/BadModulesFromClusters.sh ${Version}

  echo -e "\n...Running SummaryTable.sh $Version"
  ${LOCALHOME}/macros/SummaryTable.sh ${Version}

done

echo -e "\n...Creating Summaries"
${LOCALHOME}/getSummary.sh 

rm -f ${LOCALHOME}/lock_summ
