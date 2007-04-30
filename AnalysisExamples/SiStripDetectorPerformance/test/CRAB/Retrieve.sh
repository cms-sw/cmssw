#!/bin/sh
# Author M. De Mattia, marco.demattia@pd.infn.it
# 28/3/2007

##########################
# Checked Copy: it uses md5sum to check if the copy is identical to the original

function checkedcopy(){
  if [ -e $1 ]; then
    cp -r $1 $2
    ((++counter))
    echo $counter
    a=`md5sum $1 | awk '{print $1}'`
    b=`md5sum $2 | awk '{print $1}'`
    echo "${a}"
    echo "${b}"
    if [ "${a}" == "${b}" ]; then
      echo copy successfull
#      rm $1
#      break
    elif [ "${counter}" -le 10 ]; then
      echo different md5sum trying again
      checkedcopy $1 $2;
    else
      echo bad md5sum after 10 tries exiting
      break
    fi
  else
    echo File does not exist
  fi
}

# This is needed to set the counter to zero everytime the ccp is called.
function ccp(){
  export counter=0
  checkedcopy $1 $2
}
##########################

function Monitor(){
  # Take note of the scheduled jobs
#  for job_num in `cat ${log_path}/Status/$1 | awk '$2 ~ /Scheduled/ {print $1}'`
#    do
#      crab -status -c $2 > ${log_path}/Scheduled/$3_${Config}_${Flag}_${job_num}
#  done

  StoreDir=/data1/CrabAnalysis/$3/$3_${Config}_${Flag}
  echo $StoreDir

  # Get the output of complete jobs
  if [ `grep -c Cleared ${log_path}/Status/$1` -ne 0 ]; then
    for job_num in `cat ${log_path}/Status/$1 | awk '$2 ~ /Cleared/ {print $1}'`
      do
#      result=`cat ${log_path}/Status/$1 | awk '$2 ~ /Cleared/ {print $3}'`
#      if [ "${result}" == "(Success)" ]; then
        for File in `ls $2/res/ | grep "root"`
          do
          echo copying root file in $2/res/${File} to ${StoreDir}/${File}
          ccp $2/res/${File} ${StoreDir}/${File}
        done
        for File in `ls $2/res/ | grep "ps"`
          do
          echo copying ps file in $2/res/${File} to ${StoreDir}/${File}
          ccp $2/res/${File} ${StoreDir}/${File}
        done
#      fi
    done
  fi
}

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

#cd /analysis/sw/CRAB/CMSSW_1_3_0_pre5_v1/src/
#eval `scramv1 runtime -sh`

#export X509_USER_PROXY=`cat /analysis/sw/CRAB/log/X509_USER_PROXY.txt`
#export X509_USER_PROXY=~/public/x509up_u405
#export X509_VOMS_DIR=`cat /analysis/sw/CRAB/log/X509_VOMS_DIR.txt`
#export X509_CERT_DIR=`cat /analysis/sw/CRAB/log/X509_CERT_DIR.txt`

#echo $X509_USER_PROXY
#echo $X509_VOMS_DIR
#echo $X509_CERT_DIR

export local_crab_path=/analysis/sw/CRAB
export cfg_path=${local_crab_path}/cfg
export template_path=${local_crab_path}/templates
export log_path=${local_crab_path}/log

#while true;
#  do

echo ${log_path}

  for FileName in `ls ${log_path}/Submitted`
    do
    export Type=`echo $FileName | awk -F_ '{print $1}'`
    export Flag=`echo $FileName | awk -F_ '{print $3}'`
    export Config=`echo $FileName | awk -F_ '{print $2}'`

#    echo ${Flag}
#    echo ${Config}

    Working_CA_path=/tmp/$USER/CRAB_ClusterAnalysis_${Config}_${Flag}
    Working_TIFN_path=/tmp/$USER/CRAB_TIFNtupleMaker_${Config}_${Flag}

    log_CA=ClusterAnalysis_${Config}_${Flag}.txt
    log_TIFN=TIFNtupleMaker_${Config}_${Flag}.txt

    ##########
    ## MAIN ##
    ##########

#    if [ ${Type} == "ClusterAnalysis" ]; then
#      echo Checking status of ${FileName}
#      crab -status -c ${Working_CA_path} > ${log_path}/Status/${log_CA}
#    fi
#    if [ ${Type} == "TIFNtupleMaker" ]; then
#      echo Checking status of ${FileName}
#      crab -status -c ${Working_TIFN_path} > ${log_path}/Status/${log_TIFN}
#    fi
#    echo Status checked

#  if test -e !${Working_CA_path} -o -e !${log_path} ; then
#    echo exists
#    elif
#      echo does not exist
#  fi

#echo ${log_CA}

    # Call the monitoring function
    Monitor ${log_CA} ${Working_CA_path} "ClusterAnalysis"
    Monitor ${log_TIFN} ${Working_TIFN_path} "TIFNtupleMaker"

#    cat ${log_path}/Status/${log_CA}
#    cat ${log_path}/Status/${log_TIFN}

  done

# Wait for 1800 seconds
#  sleep 1800
#done
