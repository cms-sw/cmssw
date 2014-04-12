#!/bin/sh

run=$1

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
SELECT B.TSCKEY,
       A.GT_KEY,
       C.GT_RUN_SETTINGS_FK
FROM   CMS_TRG_L1_CONF.TRIGGERSUP_CONF A,
       CMS_WBM.RUNSUMMARY B,
       CMS_GT.GT_RUN_SETTINGS_KEY_HISTORY C
WHERE  A.TS_KEY = B.TSCKEY
AND    B.RUNNUMBER = C.RUN_NUMBER
AND    C.RUN_NUMBER = ${run}
;
EOF

exit
