set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlprompt ''

select RUN_NUMBER,to_char(START_TIME,'DD-MM-YYYY HH24:MI:SS') from CMS_GT_MON.GLOBAL_RUNS where RUN_NUMBER> 130000 order by RUN_NUMBER;

spool runstarttime.dat
/