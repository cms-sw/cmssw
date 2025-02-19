set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlprompt ''

select 
distinct runnumber from cms_runtime_logger.lumi_sections where beam1_stable=1 and beam2_stable=1 and runnumber>=132440 and runnumber<=149509 order by runnumber;

spool stablebeamruns2010.dat
/


