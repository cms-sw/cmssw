#!/bin/sh

# Look for the first of the next new L1 keys which has not been validated.
# Reference ~l1emulator/o2o/scripts/getRecentKeys.sh

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
SELECT ID, TSC_KEY
FROM
(
 SELECT ID, TSC_KEY
 FROM CMS_TRG_L1_CONF.L1_CONF
 WHERE ID > '${lastID}'
 AND TSC_KEY IS NOT NULL
 AND TSC_KEY NOT IN
 (
  SELECT DISTINCT TSC_KEY
  FROM CMS_TRG_L1_CONF.L1_CONF
  WHERE ID <= '${lastID}'
  AND TSC_KEY IS NOT NULL
 )
 ORDER BY ID
)
WHERE ROWNUM = 1
;
EOF

exit
