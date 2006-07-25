#!/bin/sh

###############################################################################################################
# general

set 

USAGE="Usage: `basename $0` mtcc-runnumber";
case $# in
1)
	RUNNR=$1;
	;;
*)
	echo $USAGE; exit 1;
	;;
esac

#--- definition of shell variables 
# this directory must be visible from remote batch machine
#DIR_WHERE_TO_EVAL="/afs/cern.ch/user/g/giordano/scratch0/CMSSW/TUTORIAL/CMSSW_0_8_0/" # Domenico's space
DIR_WHERE_TO_EVAL="/afs/cern.ch/user/d/dkcira/scratch0/MTCC/2006_07_23_code/CMSSW_0_8_0_pre3/" # Dorian's space
# directory where the job is run or submitted
if [ "${LS_SUBCWD+set}" = set ]; then
  LK_WKDIR="${LS_SUBCWD}" # directory where you submit in case of bsub
else
  LK_WKDIR=`pwd`          # directory where you run locally otherwise
fi
#
#TEMPLATE_CMSSW_CFG="/afs/cern.ch/user/d/dkcira/public/MTCC/2006_07_25_template/template_mtccoffline.cfg"
TEMPLATE_CMSSW_CFG="${LK_WKDIR}/template_mtccoffline.cfg"
# this directory will be created, use '/pool' for production and '/tmp' for testing
DIR_WHERE_TO_PUT_LARGE_FILES="./mtcc_${RUNNR}"
# files
CMSSW_CFG="${DIR_WHERE_TO_PUT_LARGE_FILES}/mtccoffline_${RUNNR}.cfg";
LOG_FILE="${DIR_WHERE_TO_PUT_LARGE_FILES}/mtcc_${RUNNR}.log";
POOL_OUTPUT_FILE="${DIR_WHERE_TO_PUT_LARGE_FILES}/mtcc_rec_${RUNNR}.root";
DQM_OUTPUT_FILE="${DIR_WHERE_TO_PUT_LARGE_FILES}/mtcc_dqm_${RUNNR}.root"
# have to find smth. more clever for below
CASTOR_DIR="/castor/cern.ch/cms/MTCC/data/0000${RUNNR}/A"
# need username to connect to cmsdisk0.cern.ch for asking list of files and then copying them
BATCH_USER_NAME=`whoami`
# for testing, if 0 no limit is set
MAX_FILES_TO_RUN_OVER=0;

# echo '###########################################################'
# echo 'SHELL VARIABLES'
# set
# echo '###########################################################'

###############################################################################################################
# definition of functions

#---
create_large_directory(){
  if [ -d "$DIR_WHERE_TO_PUT_LARGE_FILES" ]; then
    echo "directory ${DIR_WHERE_TO_PUT_LARGE_FILES} already exists, stopping here";
    exit 1;
  else
    echo "creating directory ${DIR_WHERE_TO_PUT_LARGE_FILES}"
    mkdir $DIR_WHERE_TO_PUT_LARGE_FILES;
  fi
}

#---
get_list_of_castor_files(){
 echo "getting list of files corresponding to run ${RUNNR}";
 if [ $MAX_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_DATA_FILES=`rfdir $CASTOR_DIR | grep '\.root' | sed 's/^.* //'`
 else
   echo "   !!! Caution !!!      limiting max. nr. of files per run to ${MAX_FILES_TO_RUN_OVER}"
   LIST_OF_DATA_FILES=`rfdir $CASTOR_DIR | head -${MAX_FILES_TO_RUN_OVER} | sed 's/^.* //'`
 fi
}

#---
get_list_of_cmsdisk0_files(){
 if [ $MAX_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_DATA_FILES=`ssh ${BATCH_USER_NAME}@cmsdisk0 'ls /data0/mtcc_test/' | grep ${RUNNR} | grep '\.root'`
 else
   echo "   !!! Caution !!!      limiting max. nr. of files per run to ${MAX_FILES_TO_RUN_OVER}"
   LIST_OF_DATA_FILES=`ssh ${BATCH_USER_NAME}@cmsdisk0.cern.ch 'ls /data0/mtcc_test/' | grep ${RUNNR} | grep '\.root' | head  -${MAX_FILES_TO_RUN_OVER}`
 fi
}

#---
copy_cmsdisk0_files_locally(){
echo "copying locally the cmsdisk0.cern.ch files";
      for rfile in $LIST_OF_DATA_FILES
      do
        scp ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_test/${rfile} ${DIR_WHERE_TO_PUT_LARGE_FILES}/${rfile}
      done
}

#---
copy_castor_files_locally(){ # this might not be necessary, if rfio: in poolsource works
echo "copying locally the castor files";
      for rfile in $LIST_OF_DATA_FILES
      do
        rfcp $CASTOR_DIR/${rfile} ${DIR_WHERE_TO_PUT_LARGE_FILES}/${rfile}
      done
}

#---
copy_pedestal_files(){
echo "copying pedestals";
PEDESTAL_DIR="/afs/cern.ch/user/d/dkcira/scratch0/MTCC/2006_07_23_code/CMSSW_0_8_0_pre3/src/DQM/SiStripMonitorCluster/test";
cp ${PEDESTAL_DIR}/insert_SiStripPedNoisesDB ${DIR_WHERE_TO_PUT_LARGE_FILES}/.
cp ${PEDESTAL_DIR}/insert_SiStripPedNoisesCatalog ${DIR_WHERE_TO_PUT_LARGE_FILES}/.
}

#---
create_cmssw_config_file(){
  if [ ! -f "$TEMPLATE_CMSSW_CFG" ]; then
    echo "file ${TEMPLATE_CMSSW_CFG} does not exist, stopping here";
    exit 1;
  fi
  if [ -f "$CMSSW_CFG" ]; then
    echo "file ${CMSSW_CFG} already exists, stopping here";
    exit 1;
  fi
# create list with full paths
  LIST_WITH_PATH="";
  for rfile in $LIST_OF_DATA_FILES
  do
    LIST_WITH_PATH="${LIST_WITH_PATH},\"castor:${CASTOR_DIR}/${rfile}\""
#    LIST_WITH_PATH="${LIST_WITH_PATH},\"file:${DIR_WHERE_TO_PUT_LARGE_FILES}/${rfile}\""
  done
  # remove first comma
  LIST_WITH_PATH=`echo $LIST_WITH_PATH | sed 's/\,//'`;
  echo "creating $CMSSW_CFG";
  touch $CMSSW_CFG;
  cat  "$TEMPLATE_CMSSW_CFG" | sed "s@SCRIPT_POOL_OUTPUT_FILE@${POOL_OUTPUT_FILE}@" | sed "s@SCRIPT_DQM_OUTPUT_FILE@${DQM_OUTPUT_FILE}@" | sed "s@SCRIPT_LIST_OF_FILES@${LIST_WITH_PATH}@" >>  ${CMSSW_CFG}
}

#---
runcms(){
  cd ${DIR_WHERE_TO_EVAL}; eval `scramv1 runtime -sh`;
  cd ${DIR_WHERE_TO_PUT_LARGE_FILES};
  touch
  echo "# *************** THE CFG FILE ${CMSSW_CFG}"
  cat ${CMSSW_CFG}
  echo "# ****************************"
  echo "running cmsRun -p ${CMSSW_CFG}"
  cmsRun  -p ${CMSSW_CFG}
  echo "cmsRun jobstatus: $?";
  ls -lh
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
   rfcp $LOG_FILE ${OUTPUT_CASTOR_DIR}/.
   rfcp $POOL_OUTPUT_FILE ${OUTPUT_CASTOR_DIR}/.
   rfcp $DQM_OUTPUT_FILE ${OUTPUT_CASTOR_DIR}/.
 fi
}


###############################################################################################################
# actual execution
ls -lh
create_large_directory;
get_list_of_castor_files;
create_cmssw_config_file;
copy_pedestal_files;
# copy_castor_files_locally;
echo "Running cmsRun. Log file: ${LOG_FILE}";
time runcms > ${LOG_FILE} 2>&1 ;
copy_output_to_castor "/castor/cern.ch/user/d/dkcira/MTCC/test/"

###############################################################################################################
