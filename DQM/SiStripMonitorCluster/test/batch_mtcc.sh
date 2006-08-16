#!/bin/sh

###############################################################################################################
# general

#set 

USAGE="Usage: `basename $0` mtcc-runnumber [mtcc-pedestal-runnr]";
case $# in
1)
	RUNNR=$1;
	;;
2)
	RUNNR=$1;
	PED_RUNNR=$2;
        echo "will create pedestals from runnr = ${PED_RUNNR}";
	;;
*)
	echo $USAGE; exit 1;
	;;
esac

# for online db access. some variables for the oracle tool of cmssw are not set anymore
LOCAL_ORACLE_ADMINDIR="/afs/cern.ch/project/oracle/admin/"

#--- definition of shell variables 
# where to look for input files, best would be castor but is not working properly
#RUN_ON_DISK0='no'
RUN_ON_DISK0='cmsdisk0'
# username to connect to cmsdisk0.cern.ch for asking list of files and copying them
BATCH_USER_NAME=`whoami`
# these have changed completely, need to find out
CASTOR_DIR="/castor/cern.ch/cms/MTCC/data/0000${RUNNR}/A"
PED_CASTOR_DIR="/castor/cern.ch/cms/MTCC/data/0000${PED_RUNNR}/A"
#
# this directory must be visible from remote batch machine
DIR_WHERE_TO_EVAL="/afs/cern.ch/user/d/dkcira/scratch0/MTCC/2006_08_14_code_090/CMSSW_0_9_0"
# if you want some example pedestals
PEDESTAL_DIR="/afs/cern.ch/user/d/dkcira/scratch0/MTCC/2006_08_14_code_090/pedestals";
# directory where the job is run or submitted
if [ "${LS_SUBCWD+set}" = set ]; then
  LK_WKDIR="${LS_SUBCWD}" # directory where you submit in case of bsub
  WWDIR="${WORKDIR}"
else
  LK_WKDIR=`pwd`          # directory where you run locally otherwise
  WWDIR=`pwd`
fi
#
# this directory will be created, use '/pool' for production and '/tmp' for testing
MTCC_OUTPUT_DIR="${WWDIR}/mtcc_${RUNNR}"
# directory where to copy locally input files, separate from above as this does not necessarely need to be recreated each time
MTCC_INPUT_DIR="${WWDIR}/InputFiles"
PED_MTCC_INPUT_DIR="${WWDIR}/InputFiles"
# pedestals
PED_CFG="${MTCC_OUTPUT_DIR}/${RUNNR}_mtccped.cfg";
PED_LOG_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_mtcc_pedestals.log";
# reconstruction
LIST_OF_CFG_FILES=""; # start empty, cfg files are added from script
GENERAL_LOG_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_reco.log"
# templates
TEMPLATE_CMSSW_CFG="${LK_WKDIR}/template_mtccoffline.cfg"
TEMPLATE_PED_CFG="${LK_WKDIR}/template_mtccped.cfg"
# for testing, if 0 no limit is set
MAX_FILES_TO_RUN_OVER=0;
# this will probably remain always 1 from now on
MAX_PED_FILES_TO_RUN_OVER=1;

# echo '###########################################################'
# echo 'SHELL VARIABLES'
# set
# echo '###########################################################'

###############################################################################################################
# definition of functions

#--- first general tests
general_checks(){
  if [ ! -f "$TEMPLATE_CMSSW_CFG" ]; then
    echo "file ${TEMPLATE_CMSSW_CFG} does not exist, stopping here";
    exit 1;
  fi
  if [ ! -f "$TEMPLATE_PED_CFG" ]; then
    echo "file ${TEMPLATE_PED_CFG} does not exist, stopping here";
    exit 1;
  fi
  echo "Using code from ${DIR_WHERE_TO_EVAL}";
}

#---
create_output_directory(){
  if [ -d "$MTCC_OUTPUT_DIR" ]; then
    echo "directory ${MTCC_OUTPUT_DIR} already exists, stopping here";
    exit 1;
  else
    echo "creating directory ${MTCC_OUTPUT_DIR}"
    mkdir $MTCC_OUTPUT_DIR;
  fi
}

#---
get_list_of_castor_files(){
 echo "getting from CASTOR the list of files corresponding to run ${RUNNR}";
 if [ $MAX_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_DATA_FILES=`rfdir $CASTOR_DIR | grep '\.root' | sed 's/^.* //'`
 else
   echo "   !!! Caution !!!      limiting max. nr. of files per run to ${MAX_FILES_TO_RUN_OVER}"
   LIST_OF_DATA_FILES=`rfdir $CASTOR_DIR | head -${MAX_FILES_TO_RUN_OVER} | sed 's/^.* //'`
 fi
 if [ "$LIST_OF_DATA_FILES" == ""   ] ;
 then
   echo "No input files found!!!!!! Stopping here";
   exit 1;
 fi
}

#---
get_list_of_pedestal_castor_files(){
 echo "getting from CASTOR the list of files corresponding to pedestal run ${PED_RUNNR}";
 if [ $MAX_PED_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_PED_DATA_FILES=`rfdir $CASTOR_DIR | grep '\.root' | sed 's/^.* //'`
 else
   echo "   !!! Caution !!!      limiting nr. of files for calculating pedestals to ${MAX_PED_FILES_TO_RUN_OVER}"
   LIST_OF_PED_DATA_FILES=`rfdir $CASTOR_DIR | head -${MAX_PED_FILES_TO_RUN_OVER} | sed 's/^.* //'`
 fi
 if [ "$LIST_OF_PED_DATA_FILES" == ""   ] ;
 then
   echo "No input files found!!!!!! Stopping here";
   exit 1;
 fi

}

#---
copy_castor_files_locally(){ # this might not be necessary, if rfio: in poolsource works
  echo "will copy locally the castor files";
  #
  if [ -d "$MTCC_INPUT_DIR" ]; then
    echo "directory ${MTCC_INPUT_DIR} already exists. copying files there";
  else
    echo "creating directory ${MTCC_INPUT_DIR}"
    mkdir $MTCC_INPUT_DIR;
  fi
  #
  for rfile in $LIST_OF_DATA_FILES
  do
    if [ -f ${MTCC_INPUT_DIR}/${rfile} ]; then
      echo " ${MTCC_INPUT_DIR}/${rfile} exists already, not copying."
    else
      echo "copying $CASTOR_DIR/${rfile} to ${MTCC_INPUT_DIR}/${rfile}"
      rfcp $CASTOR_DIR/${rfile} ${MTCC_INPUT_DIR}/${rfile}
    fi
  done
}

#---
get_list_of_cmsdisk0_files(){
 echo "getting from cmsdisk0.cern.ch the list of files corresponding to run ${RUNNR}";
 if [ $MAX_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_DATA_FILES=`ssh -n ${BATCH_USER_NAME}@cmsdisk0 'ls /data0/mtcc_0_9_0/' | grep ${RUNNR} | grep '\.dat'`
 else
   echo "   !!! Caution !!!      limiting max. nr. of files per run to ${MAX_FILES_TO_RUN_OVER}"
   LIST_OF_DATA_FILES=`ssh -n ${BATCH_USER_NAME}@cmsdisk0.cern.ch 'ls /data0/mtcc_0_9_0/' | grep ${RUNNR} | grep '\.dat' | head  -${MAX_FILES_TO_RUN_OVER}`
 fi
 if [ "$LIST_OF_DATA_FILES" == ""   ] ;
 then
   echo "No input files found!!!!!! Stopping here";
   exit 1;
 fi
}

#---
copy_cmsdisk0_files_locally(){
  echo "will copy locally the cmsdisk0.cern.ch files";
  #
  if [ -d "$MTCC_INPUT_DIR" ]; then
    echo "directory ${MTCC_INPUT_DIR} already exists. copying files there";
  else
    echo "creating directory ${MTCC_INPUT_DIR}"
    mkdir $MTCC_INPUT_DIR;
  fi
  #
  for rfile in $LIST_OF_DATA_FILES
  do
    if [ -f ${MTCC_INPUT_DIR}/${rfile} ]; then
      echo " ${MTCC_INPUT_DIR}/${rfile} exists already, not copying."
    else
      echo "copying  ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} to ${MTCC_INPUT_DIR}/${rfile}"
      scp ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} ${MTCC_INPUT_DIR}/${rfile}
    fi
  done
}

#---
get_list_of_cmsdisk0_pedestal_files(){
 echo "getting from cmsdisk0.cern.ch the list of files corresponding to pedestal run ${PED_RUNNR}";
 if [ $MAX_PED_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_PED_DATA_FILES=`ssh -n ${BATCH_USER_NAME}@cmsdisk0 'ls /data0/mtcc_0_9_0/' | grep ${PED_RUNNR} | grep '\.dat'`
 else
   echo "   !!! Caution !!!      limiting max. nr. of pedestal files per run to ${MAX_PED_FILES_TO_RUN_OVER}"
   LIST_OF_PED_DATA_FILES=`ssh -n ${BATCH_USER_NAME}@cmsdisk0.cern.ch 'ls /data0/mtcc_0_9_0/' | grep ${PED_RUNNR} | grep '\.dat' | head  -${MAX_PED_FILES_TO_RUN_OVER}`
 fi
 if [ "$LIST_OF_PED_DATA_FILES" == ""   ] ;
 then
   echo "No input files found!!!!!! Stopping here";
   exit 1;
 fi
}

#---
copy_cmsdisk0_ped_files_locally(){
  echo "will copy locally the cmsdisk0.cern.ch files";
  #
  if [ -d "$MTCC_INPUT_DIR" ]; then
    echo "directory ${MTCC_INPUT_DIR} already exists. copying files there";
  else
    echo "creating directory ${MTCC_INPUT_DIR}"
    mkdir $MTCC_INPUT_DIR;
  fi
  #
  for rfile in $LIST_OF_PED_DATA_FILES
  do
    if [ -f ${MTCC_INPUT_DIR}/${rfile} ]; then
      echo " ${MTCC_INPUT_DIR}/${rfile} exists already, not copying."
    else
      echo "copying ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} to ${MTCC_INPUT_DIR}/${rfile}"
      scp ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} ${MTCC_INPUT_DIR}/${rfile}
    fi
  done
}

#---
copy_example_pedestal_files(){
  echo "copying example pedestals from ${PEDESTAL_DIR}";
  cp ${PEDESTAL_DIR}/insert_SiStripPedNoisesDB ${MTCC_OUTPUT_DIR}/.
  cp ${PEDESTAL_DIR}/insert_SiStripPedNoisesCatalog ${MTCC_OUTPUT_DIR}/.
}

#---
create_cmssw_config_files(){
# create the cfg files according to the number of files the run is split from DAQ
  for rfile in $LIST_OF_DATA_FILES
  do
    if [ "$RUN_ON_DISK0" == "cmsdisk0" ]; then
       FILE_FULL_PATH="\"${MTCC_INPUT_DIR}/${rfile}\"" # in the case of cmsdisk0 have also to copy files locally
    else
       FILE_FULL_PATH="\"rfio:${CASTOR_DIR}/${rfile}\"" # more elegant solution in the case of CASTOR
    fi
    # cfg and log
    CMSSW_CFG="${MTCC_OUTPUT_DIR}/${RUNNR}_mtccoffline_${rfile}.cfg"
    LOG_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_mtccoffline_${rfile}.log"
    POOL_OUTPUT_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_rec_${rfile}.root"
    DQM_OUTPUT_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_dqm_${rfile}.root"
    #
    LIST_OF_CFG_FILES="${LIST_OF_CFG_FILES} ${CMSSW_CFG}"
    echo "creating $CMSSW_CFG";
    touch $CMSSW_CFG;
    cat  "$TEMPLATE_CMSSW_CFG" | sed "s@SCRIPT_POOL_OUTPUT_FILE@${POOL_OUTPUT_FILE}@" | sed "s@SCRIPT_DQM_OUTPUT_FILE@${DQM_OUTPUT_FILE}@" | sed "s@SCRIPT_LIST_OF_FILES@${FILE_FULL_PATH}@" >>  ${CMSSW_CFG}
  done
}

#---
create_pedestal_config_file(){
# create list with full paths
  PED_LIST_WITH_PATH="";
  for rfile in $LIST_OF_PED_DATA_FILES
  do
    if [ "$RUN_ON_DISK0" == "cmsdisk0" ]; then
       PED_LIST_WITH_PATH="${PED_LIST_WITH_PATH},\"${PED_MTCC_INPUT_DIR}/${rfile}\"" # in the case of cmsdisk0 have to copy files locally
    else
       PED_LIST_WITH_PATH="${PED_LIST_WITH_PATH},\"rfio:${PED_CASTOR_DIR}/${rfile}\""                 # more elegant solution in the case of CASTOR
    fi
  done
  # remove first comma
  PED_LIST_WITH_PATH=`echo $PED_LIST_WITH_PATH | sed 's/\,//'`;
  echo "creating $PED_CFG";
  touch $PED_CFG;
  cat  "$TEMPLATE_PED_CFG" | sed "s@SCRIPT_LIST_OF_FILES@${PED_LIST_WITH_PATH}@" >>  ${PED_CFG}
}

#---
runped(){
  cd ${DIR_WHERE_TO_EVAL}; eval `scramv1 runtime -sh`;
  export TNS_ADMIN=${LOCAL_ORACLE_ADMINDIR}
  export ORACLE_ADMINDIR=${LOCAL_ORACLE_ADMINDIR}
  cd ${MTCC_OUTPUT_DIR};
  echo "# ************************************************* CALCULATING THE PEDESTALS USING THE CFG FILE ${PED_CFG}"
  cat ${PED_CFG}
  cmsRun  -p ${PED_CFG}
  echo "pedestal jobstatus: $?";
  mv ${MTCC_OUTPUT_DIR}/Source_*${PED_RUNNR=}.root  ${PED_CFG}_pedestal_histograms.root
}

#---
runcms(){
  cd ${DIR_WHERE_TO_EVAL}; eval `scramv1 runtime -sh`;
  export TNS_ADMIN=${LOCAL_ORACLE_ADMINDIR}
  export ORACLE_ADMINDIR=${LOCAL_ORACLE_ADMINDIR}
  cd ${MTCC_OUTPUT_DIR};
  for I_CFG in ${LIST_OF_CFG_FILES}
  do
    echo ""
    echo "########################################################################"
    echo "###### RUNNING THE RECONSTRUCTION USING THE CFG FILE ${I_CFG}"
    cmsRun  -p ${I_CFG}
    echo "reconstruction jobstatus: $?";
    mv ${MTCC_OUTPUT_DIR}/monitor_cluster_summary.txt  ${I_CFG}_cluster_summary.txt
  done
  cp  ${MTCC_OUTPUT_DIR}/insert_SiStripPedNoisesDB  ${MTCC_OUTPUT_DIR}/${RUNNR}_insert_SiStripPedNoisesDB
  cp  ${MTCC_OUTPUT_DIR}/insert_SiStripPedNoisesCatalog  ${MTCC_OUTPUT_DIR}/${RUNNR}_insert_SiStripPedNoisesCatalog
}

#---
copy_output_to_castor(){
case $# in
1)
	OUTPUT_CASTOR_DIR="$1"
        ;;
*)
        echo "No output castor directory given, not performing copy_output_to_castor."
        ;;
esac
 # copy (some) output files to castor
 if [ $? ]; then # if above commands were successful
   echo "copying output files to $OUTPUT_CASTOR_DIR"
   for ifile in ${MTCC_OUTPUT_DIR}/${RUNNR}*
   do
    rfcp ${ifile}  ${OUTPUT_CASTOR_DIR}/.
   done
 fi
}

###############################################################################################################
# actual execution
###############################################################################################################
# GENERAL
ls -lh;
general_checks;
create_output_directory;

# PEDESTALS
if [ -n "$PED_RUNNR" ]; then
 if [ "$RUN_ON_DISK0" == "cmsdisk0" ]; then
  get_list_of_cmsdisk0_pedestal_files;
  copy_cmsdisk0_ped_files_locally;
 else
  get_list_of_pedestal_castor_files;
 fi
 create_pedestal_config_file;
 echo "Running pedestals. Log file: ${PED_LOG_FILE}";
 time runped > ${PED_LOG_FILE} 2>&1;
else
  copy_example_pedestal_files;
fi

# RECONSTRUCTION
if [ "$RUN_ON_DISK0" == "cmsdisk0" ]; then
   get_list_of_cmsdisk0_files;  
   copy_cmsdisk0_files_locally;
else
   get_list_of_castor_files;
fi
create_cmssw_config_files;
echo "Running reconstruction and monitoring. Log file: ${GENERAL_LOG_FILE}";
time runcms > ${GENERAL_LOG_FILE} 2>&1;

# FINAL TASKS
ls -lh;
#copy_output_to_castor "/castor/cern.ch/user/d/dkcira/MTCC/2006_08_16_090"
###############################################################################################################
