#!/bin/bash
# A script to execute O2O for a given subdetector and POOL-ORA object

# Check arguments
if [ ! -n "$2" ]  
then
  echo "Usage: `basename $0` <subdetector> <object>"
  echo "       subdetector:  Name of subdetector (ECAL, HCAL, CSC, etc.)"
  echo "       object:  Name of POOL-ORA object to execute O2O for"
  exit -1
fi

SUBDETECTOR=$1
OBJECT=$2

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

# Get the general setup, CMSSW, paths, etc.
echo "[INFO] Setting up the environment"
source general-runtime.sh
echo

# Subdetector-specific DB setup
# Sets SUBDETECTOR_OFFLINE_USER SUBDETECTOR_OFFLINE_PASSWORD
source $SUBDETECTOR_SETUP
SQLPLUS="sqlplus -S ${SUBDETECTOR_OFFLINE_USER}/${SUBDETECTOR_OFFLINE_PASSWORD}@${OFFLINE_DB}"

# Subdetector-specific object setup
# Sets MAPPING_FILE OBJECT_LIBRARY OBJECT_NAME OBJECT_TABLE TAG
source $OBJECT_SETUP

# Create the object payload tables
echo "[INFO]   Creating ${OBJECT_NAME} payload tables"
CMD="setup_pool_database $OBJECT_NAME
                         $OBJECT_LIBRARY
                         $OFFLINE_CONNECT
                         $MAPPING_FILE -o $O2ODIR"
echo $CMD
$CMD
echo

# Add the TIME column
echo "[INFO]   Adding TIME column to ${OBJECT_TABLE}"
SQL=`cat ${SQL_PATH}/add_time.sql | sed -e "s/<TABLE_NAME>/${OBJECT_TABLE}/;"`
echo $SQL
echo $SQL | $SQLPLUS
echo

# Create the object payload_o2o procedure
PAYLOAD_O2O=${SUBDETECTOR}_payload_o2o.sql
echo "[INFO]   Creating ${PAYLOAD_O2O}"
CMD="$SQLPLUS @${SQL_PATH}/${PAYLOAD_O2O}"
$CMD
echo

# Grant the general schema access to the payload_o2o procedure
# and to the top-level-table
echo "[INFO]   Granting ${GENERAL_DB_USER} access to ${PAYLOAD_O2O} AND ${OBJECT_TABLE}"
SEDSWITCH="s/<PAYLOAD_O2O>/${PAYLOAD_O2O}/; s/<TOP_TABLE>/${OBJECT_TABLE}/; s/<USER>/${GENERAL_DB_USER}/;"
SQL=`cat ${SQL_PATH}/grant_access.sql | sed -e "${SEDSWITCH}"`
echo $"$SQL" # weird quotes needed to output newlines
echo $"$SQL" | $SQLPLUS
echo

# Insert setup rows into O2O_SETUP
echo "[INFO]   Registering ${OBJECT_NAME} in O2O_SETUP"
SQLPLUS="sqlplus -S ${GENERAL_DB_USER}/${GENERAL_DB_PASSWORD}@${OFFLINE_DB}"
SEDSWITCH="s/<OBJECT>/${OBJECT_NAME}/; s/<SCHEMA>/${SUBDETECTOR_OFFLINE_USER}/; s/<TOP_TABLE>/${OBJECT_TABLE}/;"
SQL=`cat ${SQL_PATH}/insert_o2o_setup.sql | sed -e "${SEDSWITCH}"`
echo $SQL
echo $SQL | $SQLPLUS
echo
