#!/bin/bash

# A script to bootstrap the GENERAL schema for the CMS O2O setup

if [ ! -n "$O2O_SETUP_DIR" ]
then
  O2O_SETUP_DIR=`pwd`
fi

GENERAL_SETUP=${O2O_SETUP_DIR}/general-runtime.sh

# Get the general setup:  directories, offline-db login
echo "[INFO]   Setting up the environment"
source $GENERAL_SETUP
echo

# Create O2O support tables
echo "[INFO]  Creating O2O support tables"
SQLPLUS="sqlplus -S ${GENERAL_DB_USER}/${GENERAL_DB_PASSWORD}@${OFFLINE_DB}"
cat ${SQL_PATH}/create_o2o_setup.sql | $SQLPLUS
cat ${SQL_PATH}/create_o2o_log.sql | $SQLPLUS
echo

# Create the master_payload_o2o procedure
echo "[INFO]   Creating master_payload_o2o"
cat ${SQL_PATH}/master_payload_o2o.sql | $SQLPLUS
echo
