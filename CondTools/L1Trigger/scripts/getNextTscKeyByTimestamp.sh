#!/bin/sh

# Look for the first of the next new L1 keys which has not been validated.
# Reference ~l1emulator/o2o/scripts/getRecentKeys.sh
#
# The time format must be precise to nano seconds.
# Otherwise, the effect of creation_date > and <= would be unexpected.

lastCreationDate=$1

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
SELECT TO_CHAR(CREATION_DATE, 'YYYY.MM.DD_HH24:MI:SS_FF9'), TSC_KEY
FROM
(
 SELECT MAX(CREATION_DATE) CREATION_DATE, TSC_KEY
 FROM CMS_TRG_L1_CONF.L1_CONF
 WHERE CREATION_DATE >
 TO_TIMESTAMP('${lastCreationDate}', 'YYYY.MM.DD_HH24:MI:SS_FF9')
 AND TSC_KEY IS NOT NULL
 AND TSC_KEY NOT IN
 (
  SELECT DISTINCT TSC_KEY
  FROM CMS_TRG_L1_CONF.L1_CONF
  WHERE CREATION_DATE <=
  TO_TIMESTAMP('${lastCreationDate}', 'YYYY.MM.DD_HH24:MI:SS_FF9')
  AND TSC_KEY IS NOT NULL
 )
 GROUP BY TSC_KEY
 ORDER BY CREATION_DATE
)
WHERE ROWNUM = 1
;
EOF

exit
