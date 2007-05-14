#!/bin/sh

############
## PYTHON ##
############
export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DBS/Clients/PythonAPI
export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DLS/Client/LFCClient
export PYTHONPATH=$PYTHONPATH:${python_path}/COMP/DLS/Client/DliClient
export PATH=$PATH:${python_path}/COMP/:${python_path}/COMP/DLS/Client/LFCClient
############

#  python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCLocal_4/Writer --datasetPath=/TAC-TIBTOB-120-DAQ-EDM/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${list}

if [ ! -e ${log_path} ]; then
  mkdir -v ${log_path}
  mkdir -v ${list_path}
fi

##############################
## Extract list of new runs ##
##############################
echo Interrogating database $1

export list_temp=list_temp.txt

# to change the names for TIBTOBTEC samples in Bari
export list_temp_temp=list_temp_temp.txt
rm -f ${list_path}/${list_temp_temp}
touch ${list_path}/${list_temp_temp}

if [ $1 == "Bari" ]; then
  # To access Bari reconstructed TIBTOB runs
  python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-*-120-DAQ-EDM/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${datasets_list}
  cat ${list_path}/${datasets_list} | awk -F- '{print $2 "-" $9 "_" $10}' > ${list_path}/${list_temp}
  cat ${list_path}/${list_temp} | while read line; do
    if [ `echo ${line} | grep -c TIBTOBTEC` -ne 0 ]; then
#      echo if ${line}
      echo $line | awk -F_ '{print $3 "-" $1}' | awk -F- '{print "TIF-" $3}' >> ${list_path}/${list_temp_temp}
    elif [ "`echo ${line} | awk -F_ '{print $2}'`" == "" ]; then
      echo $line | awk -F_ '{print $1}' >> ${list_path}/${list_temp_temp}      
    elif [ `echo ${line} | grep -c -i slicetest` -ne 0 ]; then
      echo Run ${Run} is not a TIBTOB run
    else
#      echo else ${line}
      echo $line >> ${list_path}/${list_temp_temp}
    fi
  done

  cp ${list_path}/${list_temp_temp} ${list_path}/${list_temp}

  # Check this
  cat ${list_path}/${list_temp} | grep -v "Minus" > ${list_path}/${list_temp_temp}
  cp ${list_path}/${list_temp_temp} ${list_path}/${list_temp}
  ##

  rm ${list_path}/${list_temp_temp}
fi

if [ $1 == "FNAL" ]; then
  # To access FNAL reconstructed TIBTOB runs
  python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-*-RecoPass0/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/${datasets_list}
  cat ${list_path}/${datasets_list} | awk -F- '{print $2 "-" $4}' > ${list_path}/${list_temp}
fi

if [ $1 == "RAW" ]; then
  # To access RAW TIBTOB data
  python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-*-120-DAQ-EDM/RAW/*CMSSW_1_2_0* --logfile=${list_path}/${datasets_list}
  cat ${list_path}/${datasets_list} | awk -F- '{print $2 "-" $8}' > ${list_path}/${list_temp}
fi

echo Selecting runs of type $2

# Clean older lists
rm -f ${list_path}/${list}
touch ${list_path}/${list}

for type_ in `echo ${Config_}`; do
  if [ ${type_} != "All" ]; then
    if [ ${type_} == "TIBTOBTEC" ]; then
      # This is a patch for the problem of the different names of TIBTOBTEC (or TIF, or SliceTest) runs
      cat ${list_path}/${list_temp} | grep TIF >> ${list_path}/${list}
      #      cat ${list_path}/${list_temp} | grep ${type_} >> ${list_path}/${list}
    elif [ ${type_} == "TIBTOB" ]; then
      echo TIBTOB type_ = ${type_}
      cat ${list_path}/${list_temp} | grep ${type_} >> ${list_path}/${list}
#      cat ${list_path}/${list}
    else
      echo type_ = ${type_}
      cat ${list_path}/${list_temp} | grep ${type_} | grep -v "TIBTOB" >> ${list_path}/${list}
#      cat ${list_path}/${list}
    fi
  else
    cp ${list_path}/${list_temp} ${list_path}/${list}
  fi
done

# Clean temporary list
rm -f ${list_path}/${list_temp}

## TEST
#######
#export LOCALHOME=/analysis/sw/CRAB
#export local_crab_path=${LOCALHOME}

#export list_path=${LOCALHOME}
#export list=test_list.txt

#python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-TIBTOB-RecoPass0/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/test_list_FNAL.txt

#python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-*-120-DAQ-EDM/RECO/*CMSSW_1_3_0_pre6* --logfile=${list_path}/test_list_Bari.txt

#python ${local_crab_path}/dbsreadprocdataset.py --DBSAddress=MCGlobal/Writer --datasetPath=/TAC-*-120-DAQ-EDM/RAW/*CMSSW_1_2_0* --logfile=${list_path}/test_list_RAW.txt

#######

#if [ $2 == "TIBTOB" ]; then
#  cat ${list_path}/test_list_Bari.txt | grep $2 > ${list_path}/test_list_Bari.txt
#  cat ${list_path}/test_list_Bari.txt
#else
#  cat ${list_path}/test_list_Bari.txt | grep $2 | grep -v "TIBTOB" > ${list_path}/test_list_Bari.txt
#  cat ${list_path}/test_list_Bari.txt
#fi



###############################

export list_phys_tmp=list_phys_tmp.txt
#export list_phys_old=list_phys_old.txt

# Extract list of physics runs
wget -q -r "http://cmsdaq.cern.ch/cmsmon/cmsdb/servlet/RunSummaryTIF?RUN_BEGIN=$nextRun&RUN_END=1000000000&RUNMODE=PHYSIC&TEXT=1&DB=omds" -O ${list_path}/${list_phys_tmp}

# temporary patch since cmsmon is not responding
if [ `grep -c PHYSIC ${list_path}/${list_phys_tmp}` -ne 0 ]; then
#  mv ${list_path}/${list_phys} ${list_path}/${list_phys_old}
  mv ${list_path}/${list_phys_tmp} ${list_path}/${list_phys}
elif [ ! -e ${list_path}/${list_phys} ]; then
  echo old file does not exist
  echo using default from 
  cp ${local_crab_path}/log/${list_phys} ${list_path}/${list_phys}
else
  echo list of physics runs not correct
  echo using old list
#  cp ${list_path}/${list_phys_old} ${list_path}/${list_phys}

#  echo using list of all runs
#  cp ${list_path}/${list} ${list_path}/${list_phys}
fi

