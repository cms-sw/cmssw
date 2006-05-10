#!/bin/bash

# A script to bootstrap the GENERAL schema for the CMS O2O setup

# Get the general setup:  directories, offline-db login
echo "[INFO]   Setting up the environment"
source general-runtime.sh
echo

# Create O2O support tables
echo "[INFO]  Creating O2O support tables"
SQLPLUS="sqlplus -S ${GENERAL_DB_USER}/${GENERAL_DB_PASSWORD}@${OFFLINE_DB}"
$SQLPLUS @${SQL_PATH}/create_o2o_setup.sql
$SQLPLUS @${SQL_PATH}/create_o2o_log.sql
echo

# Create the master_payload_o2o procedure
echo "[INFO]   Creating master_payload_o2o"
$SQLPLUS @${SQL_PATH}/master_payload_o2o.sql
echo
