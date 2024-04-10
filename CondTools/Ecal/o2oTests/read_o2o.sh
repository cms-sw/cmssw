#! /bin/bash

source stats.sh

# DB parameters
SERVERNAME=devdb10
SCHEMA=cms_ecal
USER=cms_ecal
PASS=ecaldev05

# Set the pool environment
export POOL_AUTH_USER=$USER
export POOL_AUTH_PASSWORD=$PASS
export POOL_CATALOG=relationalcatalog_oracle://${SERVERNAME}/${SCHEMA}

# cmsRun
CONFFILE=load_o2o.cfg
COMMAND="cmsRun $CONFFILE"
runx "$COMMAND" 1 # time the command (1 trial)
