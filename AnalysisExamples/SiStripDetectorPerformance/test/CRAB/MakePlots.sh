#!/bin/sh
# Author M. De Mattia, marco.demattia@pd.infn.it
# 28/3/2007

function MakePlots(){

  StoreDir=/data1/CrabAnalysis/$1/${Version}/$1_${Config}_${Flag}/res
  echo $StoreDir

  cd ${StoreDir}

  FileNum=`ls *.root | grep -c root`

  do="true"

#  if [ -e ${StoreDir}/plots.txt ] && [ `cat plots.txt` == ${FileNum} ]; then
#    do="false"
#  fi

  if [ $do == "true" ]; then
    jobsList=`ls *.root | awk -F_ '{print $4}' | awk -F. '{print $1}' | tr '\n' - | sed -e 's/-*$//'`

    Name=$1_${Config}_${Flag}
    PSFile=$1_${Config}_${Flag}.ps

    echo Executing TIFmacro_chain on Run "${Flag}"
    echo "root -x -l -b -q '${local_crab_path}/TIFmacro_chain.C("'$listaFile'","'$PSFile'",true,true)' 1>output_root_macro 2>error_root_macro"
    root -x -l -b -q '${local_crab_path}/TIFmacro_chain.C("'$Name'","'$jobsList'","'$PSFile'",true,true)' 1>output_root_macro 2>error_root_macro

    echo ${FileNum} > plots.txt

  fi
}

function MergePlots(){

  StoreDir=/data1/CrabAnalysis/$1/${Version}/$1_${Config}_${Flag}/res
  echo $StoreDir

  cd ${StoreDir}

  FileNum=`ls *.root | grep -c root`

  do="true"

  # If there is more then one file
  if [ $FileNum -gt "1" ]; then

    jobsList=`ls *.root | awk -F_ '{print $4}' | awk -F. '{print $1}' | tr '\n' - | sed -e 's/-*$//'`

    # If the merging was not done for these files
    if [ -e ${StoreDir}/plots.txt ] && [ `cat plots.txt` == ${jobsList} ]; then
      do="false"
    fi
    if [ $do == "true" ]; then

      Name=$1_${Config}_${Flag}

      # If the merged root file already exists delete it,
      # otherwise it will be merged with the others
      if [ -e ${StoreDir}/${Name}.root ]; then
        rm -f ${StoreDir}/${Name}.root
      fi

      echo Executing Merging ClusterAnalysis hitograms for Run "${Flag}"
      echo "root -x -l -b -q '${local_crab_path}/AddHisto.C("'$listaFile'","'$PSFile'",true,true)' 1>output_root_macro 2>error_root_macro"
      root -x -l -b -q '${local_crab_path}/AddHisto.C("'$Name'","'$jobsList'")' 1>output_root_macro 2>error_root_macro

      echo ${jobsList} > plots.txt
    fi
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

cd /analysis/sw/CRAB/CMSSW_1_3_0_pre6_v1/src/
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
export template_path=${local_crab_path}/templates/${Version}
export log_path=${local_crab_path}/log/${Version}

#while true;
#  do

  for FileName in `ls ${log_path}/Submitted`
    do
    export Type=`echo $FileName | awk -F_ '{print $1}'`
    export Flag=`echo $FileName | awk -F_ '{print $3}'`
    export Config=`echo $FileName | awk -F_ '{print $2}'`

#if [ ${Flag} == "00006219" ]; then

    ##########
    ## MAIN ##
    ##########

    # Make the plots for TIFNtupleMaker
    if [ $Type == "TIFNtupleMaker" ] || [ $Type == "TIFNtupleMakerZS" ]; then
      MakePlots ${Type}
    fi

    # Merge the plots for ClusterAnalysis
    if [ $Type == "ClusterAnalysis" ]; then
      MergePlots "ClusterAnalysis"
    fi

#fi



done
