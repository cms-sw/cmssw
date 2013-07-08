set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlprompt ''

select RUNNUM,DATA_ID,COUNT(*) from CMS_LUMI_PROD.LUMISUMMARYV2 group by (RUNNUM,DATA_ID);

spool nlsperrun.dat
/

