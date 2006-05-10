#!/bin/bash

# A script to bootstrap a subdetector schema for the CMS O2O setup

# Check arguments
if [ ! -n "$1" ]  
then
  echo "Usage: `basename $0` <subdetector>"
  echo "       subdetector:  Name of subdetector (ECAL, HCAL, CSC, etc.)"
  exit -1
fi

SUBDETECTOR=$1
SUBDETECTOR_SETUP=${SUBDETECTOR}-db-setup.sh

if [ ! -f "$SUBDETECTOR_SETUP" ]
  then
    echo "ERROR:  $file not found" >&2
    exit -1
fi

# Get the general setup:  directories, offline-db login
echo "[INFO]   Setting up the environment"
source general-runtime.sh

# Get the subdetector setup:  online-db login, offline-db login
source ${SUBDETECTOR_SETUP}
echo

# Create a database link between the offline and the online subdetector DB schemas
echo "[INFO]   Creating database link"
SQLPLUS="sqlplus -S ${SUBDETECTOR_OFFLINE_USER}/${SUBDETECTOR_OFFLINE_PASSWORD}@${OFFLINE_DB}"
SEDSWITCH="s/<NAME>/${ONLINE_DB}/; s/<USER>/${SUBDETECTOR_ONLINE_USER}/; s/<PASSWORD>/${SUBDETECTOR_ONLINE_PASSWORD}/; s/<DB>/${ONLINE_DB}/;"
SQL=`cat ${SQL_PATH}/create_db_link.sql | sed -e "${SEDSWITCH}"`
echo $SQL
echo $SQL | $SQLPLUS
echo

# Create the IOV object tables
echo "[INFO]   Creating IOV object tables"
source IOV-object-setup.sh
CMD="setup_pool_database $OBJECT_NAME
                         $OBJECT_LIBRARY
                         $OFFLINE_CONNECT
                         $MAPPING_FILE -o $O2ODIR"
echo $CMD
$CMD
echo

# Create the metadata table
echo "[INFO]   Creating METADATA"
CMD="$SQLPLUS @${SQL_PATH}/create_metadata.sql"
$CMD
echo
