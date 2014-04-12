set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlprompt ''

select fill,run,MAX(COLLIDING_BUNCHES) from cms_brm_cond.lumisegment where 
COLLIDING_BUNCHES!=0 and COLLIDING_BUNCHES is not null and fill>1700 and beammode in ('SQUEEZE','ADJUST','STABLE BEAMS') group by fill,run order 
by fill,run;


spool collidingruns2011.dat
/
spool off

select fillnum,runnum from cms_lumi_prod.cmsrunsummary where fillnum>1700 and fillnum<3000 order by fillnum,runnum;

spool lumidbruns2011.dat
/
spool off