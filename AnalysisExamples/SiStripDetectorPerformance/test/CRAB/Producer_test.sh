#!/bin/sh
# Author M. De Mattia, marco.demattia@pd.infn.it
# 28/3/2007

# It takes 1 parameter: $1 = ClusterAnalysis or TIFNtupleMaker
function Production(){

  # Declaring and creating the directory in which to store everything
  StoreDir=/data1/CrabAnalysis/$1/$1_${Config}_${Flag}
  mkdir ${StoreDir}

  # Cfg generation and crab job creation
  if [ ! -e ${created_path}/$1_${Config}_${Flag} ]; then

    # Create crab cfg
    cat ${template_path}/template_$1.cfg | sed -e "s#OUTPUT_FILE#$1_${Config}_${Flag}#g" -e "s#CONFIG#${Config}#g" > ${cfg_path}/$1_${Config}_${Flag}.cfg
    cp ${cfg_path}/$1_${Config}_${Flag}.cfg ${StoreDir}

    # Create analysis cfg
    cat ${template_path}/template_crab.cfg | sed -e "s#OUTPUT_FILE#$1_${Config}_${Flag}#g" -e "s#CONFIG#${Config}#g" \
                          -e "s#DATASETPATH#${Datasetpath}#g" -e "s#ANALYSIS_CFG#${cfg_path}/$1_${Config}_${Flag}.cfg#" \
                          -e "s#WORKING_PATH#/tmp/$USER/CRAB_$1_${Config}_${Flag}#" > ${cfg_path}/crab_$1_${Config}_${Flag}.cfg
    cp ${cfg_path}/crab_$1_${Config}_${Flag}.cfg ${StoreDir}

    cd /tmp/${USER}

    # Job creation
    echo Creating $1_${Config}_${Flag} job with crab in /tmp/${USER}
    crab -create all -cfg ${cfg_path}/crab_$1_${Config}_${Flag}.cfg > ${created_path}/$1_${Config}_${Flag}
    cp ${created_path}/$1_${Config}_${Flag} ${StoreDir}/create_log.txt
    if [ `grep -c failed ${created_path}/$1_${Config}_${Flag}` != 0 ]; then
      grep -c failed ${created_path}/$1_${Config}_${Flag}
      grep failed ${created_path}/$1_${Config}_${Flag}
      echo Deleting $1_${Config}_${Flag} job with crab in /tmp/${USER}
      mv ${created_path}/$1_${Config}_${Flag} ${not_created_path}/$1_${Config}_${Flag}
      rm -r /tmp/$USER/CRAB_$1_${Config}_${Flag}
    fi
  fi
  # Job submission
  ls ${created_path}/$1_${Config}_${Flag}
  ls ${submitted_path}/$1_${Config}_${Flag}
  if [ -e ${created_path}/$1_${Config}_${Flag} ] && [ ! -e ${submitted_path}/$1_${Config}_${Flag} ] || [ `grep -c not ${submitted_path}/$1_${Config}_${Flag}` != 0 ]; then
    echo Submitting $1_${Config}_${Flag} job with crab in /tmp/${USER}
    crab -submit all -c /tmp/$USER/CRAB_$1_${Config}_${Flag} > ${submitted_path}/$1_${Config}_${Flag}
    cp ${submitted_path}/$1_${Config}_${Flag} ${StoreDir}/submission_log.txt
    cat ${submitted_path}/$1_${Config}_${Flag}
    if [ `grep -c Cannot ${created_path}/$1_${Config}_${Flag}` != 0 ] || [ `grep -c failed ${created_path}/$1_${Config}_${Flag}` != 0 ]; then
      mv ${submitted_path}/$1_${Config}_${Flag} ${not_submitted_path}/$1_${Config}_${Flag}
#      echo Deleting unsubmitted $1_${Config}_${Flag} job with crab in /tmp/${USER}
    fi
  fi
}

##########
## MAIN ##
##########

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh

########################################
## Patch to make it work with crontab ##
########################################
export MYHOME=/analysis/sw/CRAB/
source /analysis/sw/CRAB/crab.sh
########################################

export PYTHONPATH=$PYTHONPATH:${local_crab_path}/COMP/DBS/Clients/PythonAPI
export PYTHONPATH=$PYTHONPATH:${local_crab_path}/COMP/DLS/Client/LFCClient
export PYTHONPATH=$PYTHONPATH:${local_crab_path}/COMP/DLS/Client/DliClient
export PATH=$PATH:${local_crab_path}/COMP/:${local_crab_path}/COMP/DLS/Client/LFCClient

# Paths
export local_crab_path=/analysis/sw/CRAB
export cfg_path=${local_crab_path}/cfg
export template_path=${local_crab_path}/templates
export log_path=${local_crab_path}/log
export created_path=${log_path}/Created
export not_created_path=${log_path}/Not_Created
export submitted_path=${log_path}/Submitted
export not_submitted_path=${log_path}/Not_Submitted
export list_path=${local_crab_path}/log/list
export list=list_reco_CMSSW_1_3_0_pre6.txt
export list_phys=list_physics_runs.txt
export list_selected=list_selected_runs.txt

echo Doing eval in /analysis/sw/CRAB/CMSSW_1_3_0_pre5_v1
cd /analysis/sw/CRAB/CMSSW_1_3_0_pre5_v1/src/
eval `scramv1 runtime -sh`

#while true; do

#  python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCLocal_4/Writer --datasetPath=/TAC-TIBTOB-120-DAQ-EDM/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${list}

  echo Interrogating database
  # Extract list of new runs
  python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-TIBTOB-120-DAQ-EDM/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${list}

  wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=PHYSIC&TEXT=1&DB=omds" -O ${list_path}/${list_phys}

  # If the list of selected files is not provided or if the All flag is set use the list of all physics runs
  if [ -e ${list_path}/${list_selected} ]; then
    if [ `cat ${list_path}/${list_selected} | grep -v "#" | grep All` ]; then
      list_selected=`echo ${list_phys}`
    fi
  else
    list_selected=`echo ${list_phys}`
  fi

  for Run in `cat ${list_path}/${list_selected} | awk '{print $1}' | grep -v "#"`
    do

    Datasetpath=`grep $Run ${list_path}/${list}`
#    export Dataset="/TAC-TIBTOB-120-DAQ-EDM/RECO/CMSSW_1_3_0_pre6-DIGI-RECO-Run-0006911"
#    export Datasetpath=`echo $Dataset`
    export Flag=`echo $Datasetpath | awk -F- '{print $9}'`
    export Config=`echo $Datasetpath | awk -F- '{print $2}'`

    #echo Flag = $Flag
    #echo Config = $Config

    if [ ${Datasetpath} ]; then

      echo Run ${Run} found in the database
      echo Preparing to create jobs
#      echo Datasetpath = ${Datasetpath}

      CA=ClusterAnalysis
      TIFN=TIFNtupleMaker

      # ClusterAnalysis
      #################
      Production $CA

      # TIFNtupleMaker
      ################
      Production $TIFN

  #    echo Status of the jobs
  #    crab -status -c ${Working_CA_path}
  #    crab -status -c ${Working_TIFN_path}
  #    break
    else
      echo Run ${Run} not found in the database
    fi
  done
# Wait half a hour
#  sleep 1800
#done
