#!/bin/bash
# A script to execute O2O for a given subdetector and POOL-ORA object

# Check arguments
if [ ! -n "$3" ]  
then
  echo "Usage: `basename $0` <subdetector> <object> <append>"
  echo "       subdetector:  Name of subdetector (ECAL, HCAL, CSC, etc.)"
  echo "       object:  Name of POOL-ORA object to execute O2O for"
  echo "       append:  boolean value, if true then append IOV mode is on"
  exit -1
fi

SUBDETECTOR=$1
OBJECT=$2
APPEND=$3

SUBDETECTOR_SETUP=${SUBDETECTOR}-db-setup.sh
OBJECT_SETUP=${OBJECT}-object-setup.sh

for file in $SUBDETECTOR_SETUP $OBJECT_SETUP
do
  if [ ! -f "$file" ]
    then
      echo "ERROR:  $file not found" >&2
      exit -1
  fi
done

if [ $APPEND = 1 ]
then
  APPEND="-a"
else
  APPEND=
fi

# Get the general setup, CMSSW, paths, etc.
echo "[INFO]   Setting up the environment"
source general-runtime.sh
echo

# Subdetector-specific DB setup
# Sets SUBDETECTOR_OFFLINE_USER SUBDETECTOR_OFFLINE_PASSWORD
source $SUBDETECTOR_SETUP

# Subdetector-specific object setup
# Sets MAPPING_FILE OBJECT_LIBRARY OBJECT_NAME OBJECT_TABLE TAG
source $OBJECT_SETUP

# Log the date
echo -n [`date "+%Y-%m-%d %H:%M:%S"`] >> $LOG;
T_START=`date +%s`

# Log the object
echo -n " (${OBJECT_NAME}) " >> $LOG

# Additional setup checks
if [ ! -f "$MAPPING_FILE" ]
then
  echo "ERROR:  Mapping file $MAPPING_FILE not found" >&2
  exit -1
fi

if [ $INFINITE = 1 ]
then
  INFINITE="-i"
else
  INFINITE=
fi


###
### O2O happens here
###

# Transform payload data
T1=`date +%s`
echo "[INFO]   Transferring payload objects"
echo -n "Transferring payload objects..." >> $LOG;
SQL="call master_payload_o2o('${OBJECT_NAME}');"
echo $SQL
echo $SQL | sqlplus -S ${GENERAL_DB_USER}/${GENERAL_DB_PASSWORD}@${OFFLINE_DB} 2>> $LOG
T2=`date +%s`
T_JOB=$(($T2-$T1))
echo -n "($T_JOB s)" >> $LOG;
echo

# Poolify offline objects
T1=`date +%s`
echo "[INFO]   Registering to POOL"
echo -n "Registering to POOL..." >> $LOG;
CMD="setup_pool_database $OBJECT_NAME
                         $OBJECT_LIBRARY
                         $OFFLINE_CONNECT
                         $MAPPING_FILE -o $O2ODIR"
echo $CMD
$CMD 2>> $LOG
T2=`date +%s`
T_JOB=$(($T2-$T1))
echo -n "($T_JOB s)" >> $LOG;
echo

# Assign iov
T1=`date +%s`
echo "[INFO]   Assigning IOV"
echo -n "Assigning IOV..." >> $LOG;
CMD="cmscond_build_iov -c $OFFLINE_CONNECT
                       -d $OBJECT_LIBRARY
                       -t $OBJECT_TABLE
                       -o $OBJECT_NAME
                       $APPEND $INFINITE $TAG"
echo $CMD
$CMD 2>> $LOG
T2=`date +%s`
T_JOB=$(($T2-$T1))
echo -n "($T_JOB s)" >> $LOG;
echo

# Log the duration of the O2O
T_FINISH=`date +%s`;
T_JOB=$(($T_FINISH-$T_START))
echo "Done ($T_JOB s)." >> $LOG;

tail -n1 $LOG
