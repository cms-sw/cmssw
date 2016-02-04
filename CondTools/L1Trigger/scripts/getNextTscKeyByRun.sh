#!/bin/sh

# Look for the first of the next new L1 keys which has not been validated.
# Reference ~l1emulator/o2o/scripts/getRecentKeys.sh

lastRun=$1

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
SELECT *
FROM
(
 SELECT MAX(RUNNUMBER) RUNNUMBER, TSCKEY
 FROM CMS_WBM.RUNSUMMARY
 WHERE RUNNUMBER > ${lastRun}
 AND TSCKEY IS NOT NULL
 AND TSCKEY NOT IN
 (
  SELECT DISTINCT TSCKEY
  FROM CMS_WBM.RUNSUMMARY
  WHERE RUNNUMBER <= ${lastRun}
  AND TSCKEY IS NOT NULL
 )
 GROUP BY TSCKEY
 ORDER BY RUNNUMBER
)
WHERE ROWNUM = 1
;
EOF

exit
