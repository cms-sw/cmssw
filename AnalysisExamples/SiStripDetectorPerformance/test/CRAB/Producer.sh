#!/bin/sh
# Author M. De Mattia, marco.demattia@pd.infn.it
# 20/4/2007

# It takes 1 parameter: $1 = Job name (examples are ClusterAnalysis, TIFNtupleMaker)
function Production(){

  # Declaring the name of the job
  job_name=$1_${Config}_${Flag}

  # Declaring and creating the directory in which to store everything
  StoreDir=${MainStoreDir}/$1/${Version}/${job_name}

  if [ ! -e ${StoreDir} ]; then
    mkdir -vp ${StoreDir}
    mkdir -vp ${StoreDir}/res
    mkdir -vp ${StoreDir}/logs
  fi

  # Declare crab job dir
  CrabWorkingDir=${WorkingDir}/CRAB_${Version}/CRAB_${job_name}

  # Cfg generation and crab job creation
  created_name=${created_path}/${job_name}
  if [ ! -e ${created_name} ]; then

    echo -e "\n... Create analysis cfg"
    #####################
    cat ${template_path}/template_$1.cfg | sed -e "s#OUTPUT_FILE#${job_name}#g" -e "s#CONFIG#${Config}#g" \
                                               -e "s/#${Config}//g" > ${cfg_path}/${job_name}.cfg
    cp ${cfg_path}/${job_name}.cfg ${StoreDir}/logs
    echo -e "\n... Create crab cfg"
    #################

    cat ${template_path}/template_crab.cfg | sed -e "s#OUTPUT_FILE#${job_name}#g" -e "s#CONFIG#${Config}#g" \
                          -e "s#DATASETPATH#${Datasetpath}#g" -e "s#ANALYSIS_CFG#${cfg_path}/${job_name}.cfg#" \
                          -e "s/#SE_WHITE_LIST/${SE_WHITELIST}/" \
                          -e "s#WORKING_PATH#${CrabWorkingDir}#" > ${cfg_path}/crab_${job_name}.cfg
    cp ${cfg_path}/crab_${job_name}.cfg ${StoreDir}/logs

    cd ${WorkingDir}/CRAB_${Version}

    echo -e "\n... Job creation"
    ##############
    echo Creating ${job_name} job with crab in ${WorkingDir}/CRAB_${Version}
    crab -create all -cfg ${cfg_path}/crab_${job_name}.cfg > ${created_name}
    cp ${created_name} ${StoreDir}/logs/create_log.txt
    if [ `grep -c failed ${created_name}` != 0 ]; then
      grep -c failed ${created_name}
      grep failed ${created_name}
      echo Deleting ${job_name} job with crab in ${WorkingDir}/CRAB_${Version}
      mv ${created_name} ${not_created_path}/${job_name}
      rm -r ${CrabWorkingDir}
    fi
  fi
  echo -e "\n... Job submission"
  ################
  if [ -e ${created_name} ] && [ ! -e ${submitted_path}/${job_name} ] || [ `grep -c not ${submitted_path}/${job_name}` != 0 ]; then
    echo Submitting ${job_name} job with crab in ${WorkingDir}/CRAB_${Version}
    crab -submit all -c ${CrabWorkingDir} > ${submitted_path}/${job_name} #&
    #pid=$!
    #(sleep 60; kill -9 $pid; [ "$?" != "0" ] && echo -e "\nprocess \"crab submit\" killed after 120 seconds" >> ${submitted_path}/${job_name} )
    cp ${submitted_path}/${job_name} ${StoreDir}/logs/submission_log.txt
    cat ${submitted_path}/${job_name}
    if [ `grep -c Cannot ${created_name}` != 0 ] || [ `grep -c failed ${created_name}` != 0 ] || [ `grep -c killed ${created_name}` != 0 ]; then
      mv ${submitted_path}/${job_name} ${not_submitted_path}/${job_name}
    fi
  fi
}

##########
## MAIN ##
##########

## Grid environment
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh
## Patched CRAB
source ${MYHOME}/crab.sh

# log dirs creation
#if [ ! -e ${log_path} ]; then
  mkdir -v -p ${log_path}
  mkdir -v -p ${created_path}
  mkdir -v -p ${not_created_path}
  mkdir -v -p ${submitted_path}
  mkdir -v -p ${not_submitted_path}
#  mkdir -v -p ${list_path}
  mkdir -v -p ${log_path}/Resubmitted
  mkdir -v -p ${log_path}/Scheduled
  mkdir -v -p ${log_path}/Status
  mkdir -v -p ${log_path}/Done
  mkdir -v -p ${log_path}/Crashed
#fi

echo "... cfg dir creation"
mkdir -v -p ${cfg_path}

echo Doing eval in ${CMSSW_DIR}
cd ${CMSSW_DIR}/src
eval `scramv1 runtime -sh`



#  python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCLocal_4/Writer --datasetPath=/TAC-TIBTOB-120-DAQ-EDM/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${list}

##############################
## Extract list of new runs ##
##############################
#echo Interrogating database

# To access Bari reconstructed TIBTOB runs
# python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-TIBTOB-120-DAQ-EDM/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${list}

# To access FNAL reconstructed TIBTOB runs
# python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-TIBTOB-RecoPass0/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${list}

# To access RAW TIBTOB data
#python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-TIBTOB-120-DAQ-EDM/RAW/*CMSSW_1_2_0* --logfile=${list_path}/${list}

###############################

# Extract list of physics runs
#wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=PHYSIC&TEXT=1&DB=omds" -O ${list_path}/${list_phys}

# temporary patch since cmsmon is not responding
#if [ "`cat ${list_path}/${list_phys}`" == "" ]; then
#  echo list of physics runs is empty
#  echo using list of all runs
#  cp ${list_path}/${list} ${list_path}/${list_phys}
#fi

# If All is used set RunsList to all physics runs
if [ "`echo ${RunsList} | awk '{print $1}'`" == "All" ]; then
  # clean the RUN word
  export RunsList=`cat ${list_path}/${list_phys} | grep -v "RUN" | awk '{print $1}'`
fi

# Analyzers Names
#################
for AnalyzerName in `echo ${AnalyzersList}`; do

  # Analyzer dir creation
  if [ ! -e ${LOCALHOME}/${AnalyzerName} ]; then
    mkdir -v ${LOCALHOME}/${AnalyzerName}
  fi

  # Version dir creation
  if [ ! -e ${WorkingDir}/CRAB_${Version} ]; then
    echo Creating WorkingDir ${WorkingDir}/CRAB_${Version}
  fi
  mkdir -v -p ${WorkingDir}/CRAB_${Version}

  echo RunsList = $RunsList

  # Loop on runs
  for Run in `echo ${RunsList}`; do

    Datasetpath=`grep -i $Run ${list_path}/${datasets_list}`
    FlagConfig=`grep -i $Run ${list_path}/${list}`
    export Flag=`echo $FlagConfig | awk -F- '{print $2}'`
    export Config=`echo $FlagConfig | awk -F- '{print $1}'`

    if [ "${Datasetpath}" != "" ] && [ `grep -c ${Run} ${list_path}/${list}` -eq 1 ]; then



      echo Run ${Run} found in the database
      echo Preparing to create jobs

      ## Create and submit jobs
      #########################

      echo "... Doing Production ${AnalyzerName}"
      Production ${AnalyzerName}

    else
      echo Run ${Run} not found in the database
    fi
  done
done
