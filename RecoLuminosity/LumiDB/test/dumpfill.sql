set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlprompt ''

select 
'"'||lhcfill||'","'||injectionscheme||'","'||ncollidingbunches||'"' from cms_runtime_logger.runtime_summary where injectionscheme is not null and injectionscheme!='no_value' order by lhcfill ;

spool fillsummary.dat
/


