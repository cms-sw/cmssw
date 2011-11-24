set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlprompt ''

select 
 runnum,cmslsnum,trgcount,prescale from cms_lumi_prod.trg where runnum=182257 and bitnum=0 order by cmslsnum;

spool Algo0_182257.dat
/


