#!/bin/sh
# Author M. De Mattia, marco.demattia@pd.infn.it
# 20/4/2007

##########################
# Checked Copy: it uses md5sum to check if the copy is identical to the original

function checkedcopy(){
  if [ -e $1 ]; then
    cp -r $1 $2
    ((++counter))
    a=`md5sum $1 | awk '{print $1}'`
    b=`md5sum $2 | awk '{print $1}'`
    echo "${a}"
    echo "${b}"
    if [ "${a}" == "${b}" ]; then
      echo copy successfull removing $1
      rm -f $1
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

  # Declare name of the job
  job_name=${Type}_${Config}_${Flag}

  # Dir where to store everything
  StoreDir=${MainStoreDir}/${Type}/${Version}/${job_name}

  # Crab job dir
  CrabWorkingDir=${WorkingDir}/CRAB_${Version}/CRAB_${job_name}

  # Take note of the scheduled jobs
  if [ -e ${log_path}/Status/${job_name}.txt ]; then
    for job_num in `cat ${log_path}/Status/${job_name}.txt | awk '$2 ~ /Scheduled/ {print $1}'`
      do
        cp ${log_path}/Status/${job_name}.txt ${log_path}/Scheduled/${job_name}_${job_num}
        cp ${log_path}/Scheduled/${job_name}_${job_num} ${StoreDir}/logs/status_${job_name}.txt
    done
  fi

  echo Monitoring ${Type} job for run ${Flag} - ${Config}

  #echo "crab -status -c ${CrabWorkingDir} > ${log_path}/Status/${job_name}.txt"
  crab -status -c ${CrabWorkingDir} > ${log_path}/Status/${job_name}.txt
  cp ${log_path}/Status/${job_name}.txt ${StoreDir}/logs/status_${job_name}.txt

  # Resubmit aborted jobs only once
  #################################
  if [ `grep -c Aborted ${log_path}/Status/${job_name}.txt` -ne 0 ]; then
    for job_num in `cat ${log_path}/Status/${job_name}.txt | grep Aborted | awk '{print $1}'`; do
      if [ ! -e ${log_path}/Resubmitted/Grid_${job_name}_${job_num}.txt ]; then
        crab -resubmit ${job_num} -c ${CrabWorkingDir} > ${log_path}/Resubmitted/Grid_${job_name}_${job_num}.txt
      else
        cp ${CrabWorkingDir}/log/crab.log ${StoreDir}/logs/crab.log
      fi
    done
  fi

  # Get the output of complete jobs
  #################################
  if [ `grep -c Done ${log_path}/Status/${job_name}.txt` -ne 0 ]; then
    for num_and_status in `cat ${log_path}/Status/${job_name}.txt | awk '$2 ~ /Done/ {print $1 $3}'`
      do
      job_num=`echo $num_and_status | awk -F"(" '{print $1}'`
      result=`echo $num_and_status | awk -F"(" '{print $2}'`
      if [ $result == "Success)" ]; then
        # If it was not done or crashed yet then get the output
        #######################################################
        echo retrieving job number = "${job_num}"
        if [ ! -e ${local_crab_path}/log/Done/${job_name}_${job_num} ] && [ ! -e ${local_crab_path}/log/Crash/${job_name}_${job_num} ]; then
          crab -getoutput ${job_num} -c ${CrabWorkingDir}
          crab -status -c ${CrabWorkingDir} > ${log_path}/Done/${job_name}_${job_num}
          cp ${log_path}/Done/${job_name}_${job_num} ${StoreDir}/logs/status_${job_name}.txt
          cat ${log_path}/Done/${job_name}_${job_num} | mail -s "Output Retrieved ${Type} for job number ${job_num}" marco.de.mattia@cern.ch
          # If it has crashed
          ###################
          for zeros in `cat ${log_path}/Done/${job_name}_${job_num} | awk '$2 ~ /Cleared/ {print $4 $5}'`
            do
            if [ ${zeros} != "00" ]; then
              mv ${log_path}/Done/${job_name}_${job_num} ${log_path}/Crashed/${job_name}_${job_num}
              if [ ! -e ${log_path}/Resubmitted/${job_name}_${job_num}.txt ]; then
                crab -resubmit ${job_num} -c ${CrabWorkingDir} > ${log_path}/Resubmitted/${job_name}_${job_num}.txt
              else
                cp ${CrabWorkingDir}/log/crab.log ${StoreDir}/logs/crab.log
              fi
            # If it was completed successfully copy the output in the corresponding dir
            ###########################################################################
            elif [ ${zeros} == "00" ]; then
              echo copying root and ps files to "${StoreDir}"
              File_root=${job_name}_${job_num}.root
              File_ps=${job_name}_${job_num}.ps
              ccp ${CrabWorkingDir}/res/${File_root} ${StoreDir}/res/${File_root}
              ccp ${CrabWorkingDir}/res/${File_ps} ${StoreDir}/res/${File_ps}
            fi
          done
        fi
      fi
    done
  fi
}

##########
## MAIN ##
##########

## Grid environment
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh
## Patched CRAB
source ${MYHOME}/crab.sh

echo Doing eval in ${CMSSW_DIR}
cd ${CMSSW_DIR}/src
eval `scramv1 runtime -sh`

#while true;
#  do

  for FileName in `ls ${log_path}/Submitted`
    do
    export Type=`echo $FileName | awk -F_ '{print $1}'`
    export Flag=`echo $FileName | awk -F_ '{print $3}'`
    export Config=`echo $FileName | awk -F_ '{print $2}'`
    export SubFlag=`echo $FileName | awk -F_ '{print $4}'`

    if [ `echo ${SubFlag}` != "" ]; then
      Flag=${Flag}"_"${SubFlag}
    fi

    ##########
    ## MAIN ##
    ##########

    # Call the monitoring function
    Monitor

#  done
#sleep 3600
done
