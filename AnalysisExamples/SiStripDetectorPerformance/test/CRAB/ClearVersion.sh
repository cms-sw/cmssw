#!/bin/sh
# Author M. De Mattia, marco.demattia@pd.infn.it
# 28/3/2007

function Clear(){

  Name=$1_${Config}_${Flag}
  StoreDir=/data1/CrabAnalysis/$1/${Version}/${Name}
  echo Erasing jobs for run ${Flag}

  echo Killing remaining jobs

  # First kill submitted jobs
  if [ -e  ${log_path}/Submitted/${Name} ]; then
      echo crab -kill all -c $2
      crab -kill all -c $2
  fi

  echo Deleting directories

  if [ "$2" == "all" ]; then
      rm -v -r -f $2
      rm -v -r -f ${StoreDir}
      rm -v -r -f ${log_path}/Created/${Name}
      rm -v -r -f ${log_path}/Submitted/${Name}
      rm -v -r -f ${log_path}/Not_Created/${Name}
      rm -v -r -f ${log_path}/Not_Submitted/${Name}
      rm -v -r -f ${log_path}/Status/${Name}
      rm -v -r -f ${log_path}/Scheduled/${Name}
      rm -v -r -f ${log_path}/Done/${Name}
      rm -v -r -f ${log_path}/Cleared/${Name}
      rm -v -r -f ${log_path}/Crashed/${Name}
  fi
}

######################
## MAIN
#####################

[ "$1" == "" ] && echo "Please specify the version to be removed " && exit
[ "$2" == "" ] && echo "Please specify if only <crab> or <all>  " && exit

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

cd /analysis/sw/CRAB/CMSSW/CMSSW_1_3_0/src/
eval `scramv1 runtime -sh`
cd -

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh

########################################
## Patch to make it work with crontab ##
###########################################
export MYHOME=/analysis/sw/CRAB
# the scritps will source ${MYHOME}/crab.sh
source ${MYHOME}/crab.sh
########################################

export Version=$1

export local_crab_path=/analysis/sw/CRAB
export cfg_path=${local_crab_path}/cfg
export template_path=${local_crab_path}/templates
export log_path=${local_crab_path}/log/${Version}

echo ${log_path}

for FileName in `ls ${log_path}/Created`
  do
  export Type=`echo $FileName | awk -F_ '{print $1}'`
  export Flag=`echo $FileName | awk -F_ '{print $3}'`
  export Config=`echo $FileName | awk -F_ '{print $2}'`

  Working_path=/tmp/$USER/CRAB_${Version}/CRAB_${Type}_${Config}_${Flag}

  ##########
  ## MAIN ##
  ##########

  # Call the Clear function
  Clear ${Type} ${Working_path}
  
  [ "$2" == "all" ] && rm -vfr /data1/CrabAnalysis/${Type}/${Version}
done

[ "$2" == "all" ] && rm -vfr /data1/CrabAnalysis/logs/${Version}
[ "$2" == "all" ] && rm -vrf /data1/CrabAnalysis/*/${Version}
[ "$2" == "all" ] && rm -vrf ${log_path}