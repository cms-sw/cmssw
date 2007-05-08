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
    crab -kill all -c $2
  fi

  echo Deleting directories

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
}

######################
## MAIN
#####################

[ "$1" == "" ] && echo "Please specify the version to be removed " && exit


#source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh
#source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh
#source ~/public/crab.sh

#source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh

########################################
## Patch to make it work with crontab ##
########################################
#export MYHOME=/analysis/sw/CRAB/
#export MYHOME="${HOME}"
#source /analysis/sw/CRAB/crab.sh
########################################

cd /analysis/sw/CRAB/CMSSW/CMSSW_1_3_0/src/
eval `scramv1 runtime -sh`

#export X509_USER_PROXY=`cat /analysis/sw/CRAB/log/X509_USER_PROXY.txt`
#export X509_USER_PROXY=~/public/x509up_u405
#export X509_VOMS_DIR=`cat /analysis/sw/CRAB/log/X509_VOMS_DIR.txt`
#export X509_CERT_DIR=`cat /analysis/sw/CRAB/log/X509_CERT_DIR.txt`

#echo $X509_USER_PROXY
#echo $X509_VOMS_DIR
#echo $X509_CERT_DIR

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

  rm -vfr /data1/CrabAnalysis/${Type}/${Version}
done

rm -vfr /data1/CrabAnalysis/logs/${Version}
rm -vrf /data1/CrabAnalysis/*/${Version}
rm -vrf ${log_path}