#!/bin/sh

# List the Valid TSC Keys in OMDS

lastID=$1

DB="cms_omds_lb"
USER="cms_trg_r"
PASSWORD_FILE=/nfshome0/centraltspro/secure/$USER.txt
PASSWORD=`cat $PASSWORD_FILE`

sqlplus -S ${USER}/${PASSWORD}@${DB} <<EOF
set linesize 500
set wrap on
set heading off
set pagesize 0
set feedback off
select TSC_KEY from cms_l1_hlt.L1_HLT_CONF_LATEST_VALID_VIEW;
EOF

exit
