set colsep ','
set echo off
set feedback off
set pagesize 0
set trimspool on
set headsep off
set linesize 300
set sqlpromt ''
select '"'|| p.pathid||'","'||p.name||'"' from cms_hlt.configurations c,cms_hlt.configurationpathassoc cpa,cms_hlt.paths p,cms_hlt.pathstreamdatasetassoc psda,cms_hlt.primarydatasets pds where c.configid=cpa.configid and p.pathid=cpa.pathid and psda.pathid=p.pathid and pds.datasetid=psda.datasetid and c.configdescriptor='/cdaq/physics/Run2010/v2.3/HLT/V2' and pds.datasetlabel='MinimumBias' group by p.pathid,p.name order by p.name;

spool datasettriggers.dat
/
