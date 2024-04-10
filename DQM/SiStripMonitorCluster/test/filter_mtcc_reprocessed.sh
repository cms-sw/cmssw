#!/bin/sh
#dkcira - 2006.08.20, add more flexibility, change to options instead of add more flexibility, positional parameters

###############################################################################################################
# users can change these to their default directories that will overwrite command line options
###############################################################################################################
DEFAULT_INPUT_DIRECTORY="/castor/cern.ch/cms/testbeam/tkmtcc/P5_data/tracker_reprocessing/pass1";
#DEFAULT_CASTOR_OUTPUT_DIRECTORY="/castor/cern.ch/user/d/dkcira/MTCC/2006_08_19_filter"

###############################################################################################################
# read options
###############################################################################################################
echo $0 $@; # echo $OPTIND;
OPTERROR=11; # error option to use when exiting
#
show_usage(){
  echo ""
  echo "Usage:"
  echo "`basename $0` -r mtcc_runnumber [ -i which_input_directory -o output_directory_in_castor -h ]";
  echo ""
  echo " -h"
  echo "     print out this message"
  echo " -r  mtcc_runnumber"
  echo "      run number you want to filter"
  echo " -i which_input_directory"
  echo "      Set this to castor directory or local directory"
#  echo "      If DEFAULT_CASTOR_INPUT_DIRECTORY is set in the script, it will ignore this option"
  echo " -o output_directory_in_castor :"
  echo "      Set this for your output files to be copied to castor."
#  echo "      If DEFAULT_CASTOR_OUTPUT_DIRECTORY is set in the script, it will ignore this option"
  echo ""
}
#
# Exit and complain if no argument given.
if [ -z $1 ]; then show_usage; exit $OPTERROR; fi
#
# echo "OPTIND=$OPTIND"; echo "#=$#";
while getopts ":r:i:o:h" Option
do
  case $Option in
    r)  RUNNR=$OPTARG; echo "will reconstruct RUNNR=$RUNNR" ;;
    i)  WHICH_INPUT_DIRECTORY=$OPTARG; echo "WHICH_INPUT_DIRECTORY=$WHICH_INPUT_DIRECTORY" ;;
    o)  WHERE_TO_COPY_OUTPUT=$OPTARG; echo "WHERE_TO_COPY_OUTPUT=$WHERE_TO_COPY_OUTPUT" ;;
    h)  show_usage; exit 0;;
    *)  echo "No such option -${Option}";;
  esac
done
shift $(($OPTIND - 1)) # Decrements the argument pointer so it points to next argument.


###############################################################################################################
# define functions
###############################################################################################################
#--- this function has to be called first, obviously
set_shell_variables(){
 # this directory must be visible from remote batch machine
 DIR_WHERE_TO_EVAL="/afs/cern.ch/user/v/vciulli/scratch0/MTCC/2006_10_11_102/CMSSW_1_0_2"
 # for online db access. some variables for the oracle tool of cmssw are not set anymore
 LOCAL_ORACLE_ADMINDIR="/afs/cern.ch/project/oracle/admin/"
 # username to connect to cmsdisk0.cern.ch for asking list of files and copying them
 BATCH_USER_NAME=`whoami`
 # directory where the job is run or submitted
 if [ "${LS_SUBCWD+set}" = set ]; then
   LK_WKDIR="${LS_SUBCWD}" # directory where you submit in case of bsub
   WWDIR="${WORKDIR}"
 else
   LK_WKDIR=`pwd`          # directory where you run locally otherwise
   WWDIR=`pwd`
 fi
 # this directory will be created, use '/pool' for production and '/tmp' for testing
 MTCC_OUTPUT_DIR="${WWDIR}/mtcc_filter_${RUNNR}"
 # template
 TEMPLATE_FILTER_CFG="${LK_WKDIR}/template_filter_reprocessed.cfg"
 # config files
 FILTER_CFG="${MTCC_OUTPUT_DIR}/${RUNNR}_filter.cfg";
 # log files
 FILTER_LOG="${MTCC_OUTPUT_DIR}/${RUNNR}_filter.log";
 # histograms + pool
 FILTER_POOL_OUTPUT_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_filter_rec.root";
 DQM_OUTPUT_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_filter_dqm.root"
 # for testing, set to 100000
 MAX_FILES_TO_RUN_OVER=100000;
 # echo '###########################################################'
 # echo 'SHELL VARIABLES'; set; export;
 # echo '###########################################################'
}


#--- first general tests
inital_checks_and_settings(){
  # need a run
  if [ "X$RUNNR" == "X" ]; then
   echo "You did not choose a run number. Stopping here!"
   exit $OPTERROR;
  fi

  # need templates
  if [ ! -f "$TEMPLATE_FILTER_CFG" ]; then echo "file ${TEMPLATE_FILTER_CFG} does not exist, stopping here"; exit $OPTERROR; fi
  if [ -f "$FILTER_CFG" ]; then echo "file ${FILTER_CFG} already exists, stopping here"; exit $OPTERROR; fi

  # first choose input directory from command line or default variable
  if [ -n "$WHICH_INPUT_DIRECTORY" ]; then
         echo "Using input directory from command line option.";
  elif [ "X$DEFAULT_INPUT_DIRECTORY" != "X" ]; then # if set and not empty
         echo "Using input directory from default variable DEFAULT_INPUT_DIRECTORY";
         WHICH_INPUT_DIRECTORY="$DEFAULT_INPUT_DIRECTORY";
  else
     echo "No input directory. Stopping here!"; exit $OPTERROR;
  fi
  # then check if castor/local and if exists
  if nsls -d "$WHICH_INPUT_DIRECTORY" >& /dev/null # Suppress default output
  then
    echo "Input directory is castor $WHICH_INPUT_DIRECTORY";
    TYPE_INPUT_DIRECTORY=1; # found on castor
  elif [ -d $WHICH_INPUT_DIRECTORY ] ; then
    # transform relative path name in absolute path name if necessary
    if [ "${InputFiles:0:1}" != "/" ] ; then # does not start with "/" so is relative path name
      WHICH_INPUT_DIRECTORY="${PWD}/${WHICH_INPUT_DIRECTORY}"; # translate into absolute path name
    fi
    echo "Input directory is local $WHICH_INPUT_DIRECTORY";
    TYPE_INPUT_DIRECTORY=2; # found locally
  else
    echo "Input directory does not exist $WHICH_INPUT_DIRECTORY"
    echo "Stopping here!"; exit $OPTERROR;
  fi

  # first choose castor directory from command line or default variable
  if [  -n "$WHERE_TO_COPY_OUTPUT" ]; then
         echo "Using castor directory from command line option";
  elif [ "X${DEFAULT_CASTOR_OUTPUT_DIRECTORY}" != "X" ]; then # if set and not empty
         echo "Using castor directory from default variable DEFAULT_CASTOR_OUTPUT_DIRECTORY";
         WHERE_TO_COPY_OUTPUT="$DEFAULT_CASTOR_OUTPUT_DIRECTORY";
  else
         echo "Output files will NOT be copied to castor.";
         CASTOROUTPUT="no"
  fi
  # then check if castor directory exists
  if [ "$CASTOROUTPUT" != "no" ] ; then
   if nsls -d "$WHERE_TO_COPY_OUTPUT" > /dev/null # Suppress default output
   then
     echo "Using $WHERE_TO_COPY_OUTPUT to copy files to castor";
   else
     echo "Directory WHERE_TO_COPY_OUTPUT=$WHERE_TO_COPY_OUTPUT does not exist on castor.";
     echo "Stopping here!"; exit $OPTERROR;
   fi
  fi

  # document which code you were using
  echo "Using code from ${DIR_WHERE_TO_EVAL}";
}

#---
create_output_directory(){
  if [ -d "$MTCC_OUTPUT_DIR" ]; then
    echo "directory ${MTCC_OUTPUT_DIR} already exists, stopping here"; exit $OPTERROR;
  else
    echo "creating directory ${MTCC_OUTPUT_DIR}"; mkdir $MTCC_OUTPUT_DIR;
  fi
}

#---
get_list_of_input_files(){
 echo "getting the list of files corresponding to run ${RUNNR}";
 if [ "$TYPE_INPUT_DIRECTORY" == "1"  ] ; then
   LIST_OF_DATA_FILES=`nsls $WHICH_INPUT_DIRECTORY | grep "${RUNNR}_reco_cluster" | grep '\.root' | head -${MAX_FILES_TO_RUN_OVER}`
 elif [ "$TYPE_INPUT_DIRECTORY" == "2"  ] ; then
   LIST_OF_DATA_FILES=`ls $WHICH_INPUT_DIRECTORY | grep "${RUNNR}_reco_cluster" | grep '\.root' | head -${MAX_FILES_TO_RUN_OVER}`
 else
  echo "No such TYPE_INPUT_DIRECTORY=$TYPE_INPUT_DIRECTORY. Stopping here!"; exit $OPTERROR;
 fi
 if [ "X$LIST_OF_DATA_FILES" == "X"  ] ; then echo "No input files found. Stopping here!"; exit $OPTERROR; fi
}

#---
create_filter_config_file(){
# create list with full paths
  LIST_WITH_PATH="";
  for rfile in $LIST_OF_DATA_FILES
  do
    if [ "$TYPE_INPUT_DIRECTORY" == "1"  ]; then # castor
       LIST_WITH_PATH="${LIST_WITH_PATH},\"rfio:${WHICH_INPUT_DIRECTORY}/${rfile}\""
    elif [ "$TYPE_INPUT_DIRECTORY" == "2"  ] ; then # local
       LIST_WITH_PATH="${LIST_WITH_PATH},\"file:${WHICH_INPUT_DIRECTORY}/${rfile}\""
    else
       echo "No such TYPE_INPUT_DIRECTORY=$TYPE_INPUT_DIRECTORY. Stopping here!"; exit $OPTERROR;
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
  cmsRun ${FILTER_CFG}
  echo "filter jobstatus: $?";
}

#---
copy_output_to_castor(){
case $# in
1) OUTPUT_CASTOR_DIR="$1" ;;
*) echo "No output castor directory given, not performing copy_output_to_castor." ;;
esac
 # copy (some) output files to castor
 # copy (some) output files to castor
 if [  "X$OUTPUT_CASTOR_DIR" != "X" ]; then
   echo "copying output files to $OUTPUT_CASTOR_DIR";
   for ifile in ${MTCC_OUTPUT_DIR}/${RUNNR}_filter*
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
set_shell_variables;
inital_checks_and_settings;
create_output_directory;
get_list_of_input_files;
create_filter_config_file;
echo "Running filter. Log file: ${FILTER_LOG}";
time runfilter > ${FILTER_LOG} 2>&1 ;
ls -lh . ${MTCC_OUTPUT_DIR}/ ;

# copy output to CASTOR if the output directory variable is set
if [ -n "WHERE_TO_COPY_OUTPUT" ]; then
   copy_output_to_castor "$WHERE_TO_COPY_OUTPUT";
fi
###############################################################################################################

