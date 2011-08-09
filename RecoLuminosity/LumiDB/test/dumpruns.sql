set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlpromt ''
select '"'|| r1.runnumber||'","'||r1.time||'","'||r2.time||'"' from cms_runinfo.runsession_parameter r1, cms_runinfo.runsession_parameter r2 where 
r1.name='CMS.LVL0:START_TIME_T' and r2.name='CMS.LVL0:STOP_TIME_T' and r1.runnumber=r2.runnumber and r1.runnumber=r2.runnumber and r1.runnumber>=172549 and r1.runnumber<=172619 order by r1.runnumber;
spool runtimes.dat
/
