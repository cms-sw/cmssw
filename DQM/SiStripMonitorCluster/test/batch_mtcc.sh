#!/bin/sh
#dkcira - 2006.08.20, add more flexibility, change to options instead of add more flexibility, positional parameters


###############################################################################################################
# users can change this to their default directory that will overwrite command line options
###############################################################################################################
#DEFAULT_CASTOR_OUTPUT_DIRECTORY="/castor/cern.ch/user/d/dkcira/MTCC/2006_08_16_090"


###############################################################################################################
# read options
###############################################################################################################
echo $0 $@; # echo $OPTIND;
OPTERROR=11; # error option to use when exiting
#
show_usage(){
  echo ""
  echo "Usage:"
  echo "`basename $0` -r mtcc_runnumber [ -w which_input_files -f first_file_of_run -l last_file_of_run -p mtcc_pedestal_runnr -o output_directory_in_castor  -t input_file_ending (default dat) -h ]";
  echo ""
  echo " -h"
  echo "     print out this message"
  echo " -r  mtcc_runnumber"
  echo "      run number you want to reconstruct"
  echo " -w which_input_files :"
  echo "      1 - copy files from castor (default option)"
  echo "      2 - run directly on castor files"
  echo "      3 - copy files from cmsdisk0"
  echo "      4 - run on local files in the subdirectory InputFiles/"
  echo " -o output_directory_in_castor :"
  echo "      Set this for your output files to be copied to castor."
  echo "      If DEFAULT_CASTOR_OUTPUT_DIRECTORY is set in the script, it will ignore this option"
  echo ""
}
# Exit and complain if no argument given.
if [ -z $1 ]; then show_usage; exit $OPTERROR; fi
#
# echo "OPTIND=$OPTIND"; echo "#=$#";
while getopts ":r:p:f:l:w:o:t:h" Option
do
  case $Option in
    r)	RUNNR=$OPTARG; echo "will reconstruct RUNNR=$RUNNR" ;;
    p)	PED_RUNNR=$OPTARG; echo "will create pedestals from PED_RUNNR=$PED_RUNNR" ;;
    f)	FIRSTFILE=$OPTARG; echo "first file of run to be reconstructed FIRSTFILE=$FIRSTFILE" ;;
    l)	LASTFILE=$OPTARG; echo "last file of run to be reconstructed LASTFILE=$LASTFILE" ;;
    w)	WHICH_INPUT_FILES=$OPTARG; echo "WHICH_INPUT_FILES=$WHICH_INPUT_FILES" ;;
    o)  WHERE_TO_COPY_OUTPUT=$OPTARG; echo "WHERE_TO_COPY_OUTPUT=$WHERE_TO_COPY_OUTPUT" ;;
    t)  INPUT_FILE_ENDING=$OPTARG; echo "INPUT_FILE_ENDING=$INPUT_FILE_ENDING" ;;
    h)	show_usage; exit 0;;
    *)	echo "No such option -${Option}";;
  esac
done
shift $(($OPTIND - 1)) # Decrements the argument pointer so it points to next argument.


###############################################################################################################
# define functions
###############################################################################################################
#--- this function has to be called first, obviously
set_shell_variables(){
 # this directory must be visible from remote batch machine
 DIR_WHERE_TO_EVAL="/afs/cern.ch/user/d/dkcira/scratch0/MTCC/2006_08_14_code_090/CMSSW_0_9_0"
 # if you want some example pedestals
 PEDESTAL_DIR="/afs/cern.ch/user/d/dkcira/scratch0/MTCC/2006_08_14_code_090/pedestals";
 # this will probably remain always 1 from now on
 MAX_PED_FILES_TO_RUN_OVER=1;
 # for online db access. some variables for the oracle tool of cmssw are not set anymore
 LOCAL_ORACLE_ADMINDIR="/afs/cern.ch/project/oracle/admin/"
 #--
 # username to connect to cmsdisk0.cern.ch for asking list of files and copying them
 BATCH_USER_NAME=`whoami`
 # these have changed completely, need to find out
 CASTOR_DIR="/castor/cern.ch/cms/MTCC/data/0000${RUNNR}/A"
 PED_CASTOR_DIR="/castor/cern.ch/cms/MTCC/data/0000${PED_RUNNR}/A"
 #
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
 GENERAL_LOG_FILE="${MTCC_OUTPUT_DIR}/${RUNNR}_reco_${FIRSTFILE}.log"
 LOCAL_INPUT_DIRECTORY="${MTCC_OUTPUT_DIR}/InputFiles";
 # templates
 TEMPLATE_CMSSW_CFG="${LK_WKDIR}/template_mtccoffline.cfg"
 TEMPLATE_PED_CFG="${LK_WKDIR}/template_mtccped.cfg"
 # echo '###########################################################'
 # echo 'SHELL VARIABLES'; set; export;
 # echo '###########################################################'
}

#--- initial checks for inconsistencies
inital_checks_and_settings(){
  # need a run
  if [ "X$RUNNR" == "X" ]; then
   echo "You did not choose a run number. Stopping here!"
   exit $OPTERROR;
  fi

  # need templates
  if [ ! -f "$TEMPLATE_CMSSW_CFG" ]; then echo "file ${TEMPLATE_CMSSW_CFG} does not exist, stopping here"; exit $OPTERROR; fi
  if [ ! -f "$TEMPLATE_PED_CFG" ]; then echo "file ${TEMPLATE_PED_CFG} does not exist, stopping here"; exit $OPTERROR; fi

  # need input files option
  if [ -z "$WHICH_INPUT_FILES" ] ; then WHICH_INPUT_FILES=1; fi
  case $WHICH_INPUT_FILES in
	1|2|3|4) echo "WHICH_INPUT_FILES=$WHICH_INPUT_FILES";;
	*) echo "no such WHICH_INPUT_FILES=$WHICH_INPUT_FILES . exit here!"; exit $OPTERROR;;
  esac

  # choose file ending, maybe will go to "root" later
  if [ -z "$INPUT_FILE_ENDING" ] ; then
     INPUT_FILE_ENDING="dat";
     echo "Using default INPUT_FILE_ENDING=$INPUT_FILE_ENDING";
  fi

  # first choose castor directory from command line or default variable
  if [ "X${DEFAULT_CASTOR_OUTPUT_DIRECTORY}" != "X" ]; then # if set and not empty
         echo "Using castor directory from default variable DEFAULT_CASTOR_OUTPUT_DIRECTORY";
         WHERE_TO_COPY_OUTPUT="$DEFAULT_CASTOR_OUTPUT_DIRECTORY";
  elif [  -n "$WHERE_TO_COPY_OUTPUT" ]; then
         echo "Using castor directory from command line option";
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
get_list_of_local_files(){
 echo "Getting list of files to be reconstructed from local directory."
 if [ -z "$FIRSTFILE" ] ; then FIRSTFILE=1  ; fi
 if [ -z "$LASTFILE" ]  ; then LASTFILE=`ls $LOCAL_INPUT_DIRECTORY | grep ${RUNNR} | grep "\.${INPUT_FILE_ENDING}" | wc -l | sed 's/[ ]*//g'`; fi
 let "HOWMANYFILES = $LASTFILE - $FIRSTFILE + 1";
 echo "FIRSTFILE=$FIRSTFILE LASTFILE=$LASTFILE HOWMANYFILES=$HOWMANYFILES"
 #
 # the funny sort is done so that the files are ordered 1, 2, 3, ..., 10, 11, ..., and not 1,10,11,...,2,20, and so on
 LIST_OF_DATA_FILES=`ls $LOCAL_INPUT_DIRECTORY |  grep ${RUNNR} | grep "\.${INPUT_FILE_ENDING}" | sed 's/^.* //' | sort -n -t . +4 | head -${LASTFILE} | tail -${HOWMANYFILES}`
 if [ "$LIST_OF_DATA_FILES" == "" ] ; then echo "No input reco files found!!!!!! Stopping here"; exit $OPTERROR; fi
}

#---
get_list_of_local_pedestal_files(){
 echo "Getting list of pedestal files from local directory."
 if [ $MAX_PED_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_PED_DATA_FILES=`ls $LOCAL_INPUT_DIRECTORY | grep "\.${INPUT_FILE_ENDING}" | sed 's/^.* //'`
 else
   echo "   !!! Caution !!!      limiting max. nr. of files per run to ${MAX_PED_FILES_TO_RUN_OVER}"
   LIST_OF_PED_DATA_FILES=`ls $LOCAL_INPUT_DIRECTORY | grep "\.${INPUT_FILE_ENDING}" | head -${MAX_PED_FILES_TO_RUN_OVER} | sed 's/^.* //'`
 fi
 if [ "$LIST_OF_PED_DATA_FILES" == "" ] ; then echo "No input pedestal files found!!!!!! Stopping here"; exit $OPTERROR; fi
}

#---
get_list_of_castor_files(){
 echo "Getting list of files to be reconstructed from castor."
 if [ -z "$FIRSTFILE" ] ; then FIRSTFILE=1  ; fi
 if [ -z "$LASTFILE" ]  ; then LASTFILE=`nsls $CASTOR_DIR | grep ${RUNNR} | grep "\.${INPUT_FILE_ENDING}" | wc -l | sed 's/[ ]*//g'`; fi
 let "HOWMANYFILES = $LASTFILE - $FIRSTFILE + 1";
 echo "FIRSTFILE=$FIRSTFILE LASTFILE=$LASTFILE HOWMANYFILES=$HOWMANYFILES"
 echo "getting from CASTOR the list of files corresponding to run ${RUNNR}";
 LIST_OF_DATA_FILES=`nsls $CASTOR_DIR | grep ${RUNNR} | grep "\.${INPUT_FILE_ENDING}" | sed 's/^.* //' | sort -n -t . +4 | head -${LASTFILE} | tail -${HOWMANYFILES}`
 if [ "$LIST_OF_DATA_FILES" == "" ] ; then echo "No input files found!!!!!! Stopping here"; exit $OPTERROR; fi
}

#---
get_list_of_pedestal_castor_files(){
 echo "Getting list of pedestal files to be reconstructed from castor."
 if [ $MAX_PED_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_PED_DATA_FILES=`nsls $CASTOR_DIR | grep "\.${INPUT_FILE_ENDING}" | sed 's/^.* //'`
 else
   echo "   !!! Caution !!!      limiting nr. of files for calculating pedestals to ${MAX_PED_FILES_TO_RUN_OVER}"
   LIST_OF_PED_DATA_FILES=`nsls $CASTOR_DIR | grep "\.${INPUT_FILE_ENDING}" | head -${MAX_PED_FILES_TO_RUN_OVER} | sed 's/^.* //'`
 fi
 if [ "$LIST_OF_PED_DATA_FILES" == "" ] ; then echo "No input pedestal files found!!!!!! Stopping here"; exit $OPTERROR; fi
}

#---
copy_castor_files_locally(){
  echo "LIST_OF_DATA_FILES="; echo "$LIST_OF_DATA_FILES";
  echo "Will copy locally the input files from castor.";
  if [ -d "$MTCC_INPUT_DIR" ]; then
    echo "directory ${MTCC_INPUT_DIR} already exists. copying files there";
  else
    echo "creating directory ${MTCC_INPUT_DIR}"; mkdir $MTCC_INPUT_DIR;
  fi
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
copy_castor_pedestal_files_locally(){
  echo "LIST_OF_PED_DATA_FILES="; echo "$LIST_OF_PED_DATA_FILES";
  echo "Will copy locally the pedestal files from castor.";
  if [ -d "$MTCC_INPUT_DIR" ]; then
    echo "directory ${MTCC_INPUT_DIR} already exists. copying files there";
  else
    echo "creating directory ${MTCC_INPUT_DIR}"
    mkdir $MTCC_INPUT_DIR;
  fi
  for rfile in $LIST_OF_PED_DATA_FILES
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
 echo "Getting list of files to be reconstructed from cmsdisk0."
 if [ -z "$FIRSTFILE" ] ; then FIRSTFILE=1  ; fi
 if [ -z "$LASTFILE" ]  ; then LASTFILE=`ssh -n ${BATCH_USER_NAME}@cmsdisk0 'ls /data0/mtcc_0_9_0/' | grep ${RUNNR} | grep "\.${INPUT_FILE_ENDING}" | wc -l | sed 's/[ ]*//g'`; fi
 let "HOWMANYFILES = $LASTFILE - $FIRSTFILE + 1";
 echo "FIRSTFILE=$FIRSTFILE LASTFILE=$LASTFILE HOWMANYFILES=$HOWMANYFILES"
 LIST_OF_DATA_FILES=`ssh -n ${BATCH_USER_NAME}@cmsdisk0 'ls /data0/mtcc_0_9_0/' | grep ${RUNNR} | grep "\.${INPUT_FILE_ENDING}" | sort -n -t . +4 | head -${LASTFILE} | tail -${HOWMANYFILES}`
 if [ "$LIST_OF_DATA_FILES" == ""   ] ; then
   echo "No input files found!!!!!! Stopping here"; exit $OPTERROR;
 fi
}

#---
copy_cmsdisk0_files_locally(){
  echo "LIST_OF_DATA_FILES=$LIST_OF_DATA_FILES"
  echo "Will copy locally the input files from cmsdisk0.";
  if [ -d "$MTCC_INPUT_DIR" ]; then
    echo "directory ${MTCC_INPUT_DIR} already exists. copying files there";
  else
    echo "creating directory ${MTCC_INPUT_DIR}"; mkdir $MTCC_INPUT_DIR;
  fi
  for rfile in $LIST_OF_DATA_FILES
  do
    if [ -f ${MTCC_INPUT_DIR}/${rfile} ]; then
      echo " ${MTCC_INPUT_DIR}/${rfile} exists already, not copying."
    else
      echo "copying  ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} to ${MTCC_INPUT_DIR}/${rfile}"
      scp -c blowfish ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} ${MTCC_INPUT_DIR}/${rfile}
    fi
  done
}

#---
get_list_of_cmsdisk0_pedestal_files(){
 echo "Getting list of pedestal files from cmsdisk0."
 if [ $MAX_PED_FILES_TO_RUN_OVER -eq 0 ]
 then
   LIST_OF_PED_DATA_FILES=`ssh -n ${BATCH_USER_NAME}@cmsdisk0 'ls /data0/mtcc_0_9_0/' | grep ${PED_RUNNR} | grep "\.${INPUT_FILE_ENDING}"`
 else
   echo "   !!! Caution !!!      limiting max. nr. of pedestal files per run to ${MAX_PED_FILES_TO_RUN_OVER}"
   LIST_OF_PED_DATA_FILES=`ssh -n ${BATCH_USER_NAME}@cmsdisk0.cern.ch 'ls /data0/mtcc_0_9_0/' | grep ${PED_RUNNR} | grep "\.${INPUT_FILE_ENDING}" | head  -${MAX_PED_FILES_TO_RUN_OVER}`
 fi
 if [ "$LIST_OF_PED_DATA_FILES" == ""   ] ;
 then
   echo "No input files found!!!!!! Stopping here";
   exit $OPTERROR;
 fi
}

#---
copy_cmsdisk0_pedestal_files_locally(){
  echo "LIST_OF_PED_DATA_FILES=$LIST_OF_PED_DATA_FILES"
  echo "Will copy locally the pedestal files from cmsdisk0.";
  if [ -d "$MTCC_INPUT_DIR" ]; then
    echo "directory ${MTCC_INPUT_DIR} already exists. copying files there";
  else
    echo "creating directory ${MTCC_INPUT_DIR}"
    mkdir $MTCC_INPUT_DIR;
  fi
  for rfile in $LIST_OF_PED_DATA_FILES
  do
    if [ -f ${MTCC_INPUT_DIR}/${rfile} ]; then
      echo " ${MTCC_INPUT_DIR}/${rfile} exists already, not copying."
    else
      echo "copying ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} to ${MTCC_INPUT_DIR}/${rfile}"
      scp -c blowfish ${BATCH_USER_NAME}@cmsdisk0.cern.ch:/data0/mtcc_0_9_0/${rfile} ${MTCC_INPUT_DIR}/${rfile}
    fi
  done
}

#---
copy_example_pedestal_files(){
  echo "Copying example pedestals from ${PEDESTAL_DIR}";
  cp ${PEDESTAL_DIR}/insert_SiStripPedNoisesDB ${MTCC_OUTPUT_DIR}/.
  cp ${PEDESTAL_DIR}/insert_SiStripPedNoisesCatalog ${MTCC_OUTPUT_DIR}/.
}

#---
create_cmssw_config_files(){
# create the cfg files according to the number of files the run is split from DAQ
  for rfile in $LIST_OF_DATA_FILES
  do
    if [ "$WHICH_INPUT_FILES" == "1" ]; then # files in castor and copy locally
      FILE_FULL_PATH="\"${MTCC_INPUT_DIR}/${rfile}\"" # files are local or will be copied locally
    elif [ "$WHICH_INPUT_FILES" == "3" ]; then # can only copy locally in case of cmsdisk0
      FILE_FULL_PATH="\"${MTCC_INPUT_DIR}/${rfile}\"" # files are local or will be copied locally
    elif [ "$WHICH_INPUT_FILES" == "4" ]; then # they have already been copied to some local directory - access them directly
      FILE_FULL_PATH="\"${MTCC_INPUT_DIR}/${rfile}\"" # files are local or will be copied locally
    elif [ "$WHICH_INPUT_FILES" == "2" ]; then # files in castor, run remotely
       FILE_FULL_PATH="\"rfio:${CASTOR_DIR}/${rfile}\"" # files are in castor and not copied
    else
       echo "Do not know what to do WHICH_INPUT_FILES=${WHICH_INPUT_FILES}";
       echo "Stopping here!."; exit $OPTERROR;
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
   if [ "$WHICH_INPUT_FILES" == "1" ]; then # files in castor and copy locally
       PED_LIST_WITH_PATH="${PED_LIST_WITH_PATH},\"${PED_MTCC_INPUT_DIR}/${rfile}\""        # local
   elif [ "$WHICH_INPUT_FILES" == "3" ]; then # can only copy locally in case of cmsdisk0
       PED_LIST_WITH_PATH="${PED_LIST_WITH_PATH},\"${PED_MTCC_INPUT_DIR}/${rfile}\""        # local
   elif [ "$WHICH_INPUT_FILES" == "4" ]; then # they have already been copied to some local directory - access them directly
       PED_LIST_WITH_PATH="${PED_LIST_WITH_PATH},\"${PED_MTCC_INPUT_DIR}/${rfile}\""        # local
   elif [ "$WHICH_INPUT_FILES" == "2" ]; then # files in castor, run remotely
       PED_LIST_WITH_PATH="${PED_LIST_WITH_PATH},\"rfio:${PED_CASTOR_DIR}/${rfile}\""       # castor
   else
       echo "Do not know what to do WHICH_INPUT_FILES=${WHICH_INPUT_FILES}";
       echo "Stopping here!."; exit $OPTERROR;
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
  cat ${PED_CFG};
  cmsRun ${PED_CFG};
  echo "pedestal jobstatus: $?";
  mv ${MTCC_OUTPUT_DIR}/Source_*${PED_RUNNR}.root  ${PED_CFG}_pedestal_histograms.root
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
    cmsRun ${I_CFG}
    echo "reconstruction jobstatus: $?";
    mv ${MTCC_OUTPUT_DIR}/monitor_cluster_summary.txt  ${I_CFG}_cluster_summary.txt
    mv ${MTCC_OUTPUT_DIR}/monitor_digi_summary.txt  ${I_CFG}_digi_summary.txt
  done
  cp  ${MTCC_OUTPUT_DIR}/insert_SiStripPedNoisesDB  ${MTCC_OUTPUT_DIR}/${RUNNR}_insert_SiStripPedNoisesDB
  cp  ${MTCC_OUTPUT_DIR}/insert_SiStripPedNoisesCatalog  ${MTCC_OUTPUT_DIR}/${RUNNR}_insert_SiStripPedNoisesCatalog
}

#---
copy_output_to_castor(){
case $# in
1) OUTPUT_CASTOR_DIR="$1" ;;
*) echo "No output castor directory given, not performing copy_output_to_castor." ;;
esac
 # copy (some) output files to castor
 if [  "X$OUTPUT_CASTOR_DIR" != "X" ]; then
   echo "copying output files to $OUTPUT_CASTOR_DIR";
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
set_shell_variables;
inital_checks_and_settings;
create_output_directory;
ls -lh;

# PEDESTALS
if [ -n "$PED_RUNNR" ]; then
 if [ "$WHICH_INPUT_FILES" == "1" ]; then # files in castor and copy locally
   get_list_of_pedestal_castor_files;
   copy_castor_pedestal_files_locally;
 elif [ "$WHICH_INPUT_FILES" == "2" ]; then # files in castor, run remotely
   get_list_of_pedestal_castor_files;
 elif [ "$WHICH_INPUT_FILES" == "3" ]; then # can only copy locally in case of cmsdisk0
   get_list_of_cmsdisk0_pedestal_files;
   copy_cmsdisk0_pedestal_files_locally;
 elif [ "$WHICH_INPUT_FILES" == "4" ]; then # they have already been copied to some local directory - access them directly
   get_list_of_local_pedestal_files;
 else
   echo "Not clear where to get files WHICH_INPUT_FILES=$WHICH_INPUT_FILES"; echo "Stopping here!"; exit $OPTERROR;
 fi
 #
 create_pedestal_config_file;
 echo "Running pedestals. Log file: ${PED_LOG_FILE}";
 time runped > ${PED_LOG_FILE} 2>&1;
else
  copy_example_pedestal_files;
fi

# RECONSTRUCTION
#WHICH_INPUT_FILES
#  echo "1 - copy files from castor (default option)"
#  echo "2 - run directly on castor files"
#  echo "3 - copy files from cmsdisk0"
#  echo "4 - run on local files in InputFiles/ subdirectory"
if [ "$WHICH_INPUT_FILES" == "1" ]; then # files in castor and copy locally
   get_list_of_castor_files;  
   copy_castor_files_locally;
elif [ "$WHICH_INPUT_FILES" == "2" ]; then # files in castor, run remotely
   get_list_of_castor_files;
elif [ "$WHICH_INPUT_FILES" == "3" ]; then # can only copy locally in case of cmsdisk0
   get_list_of_cmsdisk0_files;  
   copy_cmsdisk0_files_locally;
elif [ "$WHICH_INPUT_FILES" == "4" ]; then # they have already been copied to some local directory - access them directly
   get_list_of_local_files;
else
   echo "Not clear where to get files WHICH_INPUT_FILES=${WHICH_INPUT_FILES}"; echo "Stopping here!"; exit $OPTERROR;
fi
#
create_cmssw_config_files;
echo "Running reconstruction and monitoring. Log file: ${GENERAL_LOG_FILE}";
time runcms > ${GENERAL_LOG_FILE} 2>&1;
ls -lh . ${MTCC_OUTPUT_DIR}/ ;

# copy output to CASTOR if the output directory variable is set
if [ -n "WHERE_TO_COPY_OUTPUT" ]; then
   copy_output_to_castor "$WHERE_TO_COPY_OUTPUT";
fi

###############################################################################################################
# end
###############################################################################################################
