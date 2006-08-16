#!/bin/sh

###############################################################################################################
# general

#set 

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
RUN_ON_DISK0='no'
#RUN_ON_DISK0='cmsdisk0'
# this directory must be visible from remote batch machine
DIR_WHERE_TO_EVAL="/afs/cern.ch/user/d/dkcira/scratch0/MTCC/2006_08_14_code_090/CMSSW_0_9_0"
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
MTCC_OUTPUT_DIR="${WWDIR}/mtcc_filter_${RUNNR}"
# directory where to copy locally input files, separate from above as this does not necessarely need to be recreated each time
MTCC_INPUT_DIR="${WWDIR}/InputFiles"
#config files
FILTER_CFG="${MTCC_OUTPUT_DIR}/mtccfilter_${RUNNR}.cfg";
# log files
FILTER_LOG="${MTCC_OUTPUT_DIR}/mtcc_filter_${RUNNR}.log";
# histograms + pool
FILTER_POOL_OUTPUT_FILE="${MTCC_OUTPUT_DIR}/mtcc_filter_rec_${RUNNR}.root";
DQM_OUTPUT_FILE="${MTCC_OUTPUT_DIR}/mtcc_filter_dqm_${RUNNR}.root"
# template
TEMPLATE_FILTER_CFG="${LK_WKDIR}/template_filter.cfg"
# have to find smth. more clever for below
CASTOR_DIR="/castor/cern.ch/user/d/dkcira/MTCC/2006_08_16_090"
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

#--- first general tests
general_checks(){
  if [ ! -f "$TEMPLATE_FILTER_CFG" ]; then
    echo "file ${TEMPLATE_FILTER_CFG} does not exist, stopping here";
    exit 1;
  fi
  if [ -f "$FILTER_CFG" ]; then
    echo "file ${FILTER_CFG} already exists, stopping here";
    exit 1;
  fi
  echo "using code from ${DIR_WHERE_TO_EVAL}"
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
get_list_of_castor_reco_files(){
 echo "getting from CASTOR the list of files corresponding to run ${RUNNR}";
 LIST_OF_DATA_FILES=`rfdir $CASTOR_DIR | grep "${RUNNR}_rec" | grep '\.root' | sed 's/^.* //'`
 if [ "$LIST_OF_DATA_FILES" == ""   ] ;
 then
   echo "No input files found!!!!!! Stopping here";
   exit 1;
 fi
}

#---
create_filter_config_file(){
# create list with full paths
  LIST_WITH_PATH="";
  for rfile in $LIST_OF_DATA_FILES
  do
    if [ "$RUN_ON_DISK0" == "cmsdisk0" ]; then
       LIST_WITH_PATH="${LIST_WITH_PATH},\"file:${MTCC_INPUT_DIR}/${rfile}\"" # in the case of cmsdisk0 have to copy files locally
    else
       LIST_WITH_PATH="${LIST_WITH_PATH},\"rfio:${CASTOR_DIR}/${rfile}\""                 # more elegant solution in the case of CASTOR
    fi
  done
  # remove first comma
  LIST_WITH_PATH=`echo $LIST_WITH_PATH | sed 's/\,//'`;
  echo "creating $FILTER_CFG";
  touch $FILTER_CFG;
  cat  "$TEMPLATE_FILTER_CFG" | sed "s@SCRIPT_POOL_OUTPUT_FILE@${FILTER_POOL_OUTPUT_FILE}@" | sed "s@SCRIPT_DQM_OUTPUT_FILE@${DQM_OUTPUT_FILE}@" | sed "s@SCRIPT_LIST_OF_FILES@${LIST_WITH_PATH}@" >>  ${FILTER_CFG}
}

#---
runfilter(){
  cd ${DIR_WHERE_TO_EVAL}; eval `scramv1 runtime -sh`;
  cd ${MTCC_OUTPUT_DIR};
  echo "##### RUNNING THE RECONSTRUCTION USING THE CFG FILE ${FILTER_CFG}"
  cat ${FILTER_CFG}
  cmsRun  -p ${FILTER_CFG}
  echo "filter jobstatus: $?";
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
   rfcp $FILTER_CFG ${OUTPUT_CASTOR_DIR}/.
   rfcp $FILTER_LOG ${OUTPUT_CASTOR_DIR}/.
   rfcp $FILTER_POOL_OUTPUT_FILE ${OUTPUT_CASTOR_DIR}/.
   rfcp $DQM_OUTPUT_FILE ${OUTPUT_CASTOR_DIR}/.
 fi
}

###############################################################################################################
# actual execution
###############################################################################################################
# GENERAL
ls -lh;
general_checks;
create_output_directory;
get_list_of_castor_reco_files;
create_filter_config_file;
echo "Running filter. Log file: ${FILTER_LOG}";
time runfilter > ${FILTER_LOG} 2>&1 ;

# FINAL TASKS
ls -lh;
#copy_output_to_castor "/castor/cern.ch/user/d/dkcira/MTCC/filter"
###############################################################################################################

