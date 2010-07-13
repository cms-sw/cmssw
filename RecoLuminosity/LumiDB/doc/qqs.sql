select fillnum,sequence,hltkey,startime,stoptime from cmsrunsummary where runnum=:runnum
select cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenery from lumisummary where runnum=:runnum and lumiversion=:lumiversion
select sum(instlumi) from lumisummary where runnum=:runnum and lumiversion=:lumiversion
select cmslsnum,trgcount,deadtime,bitname,prescale from trg where runnum=:runnum and bitnum=0;
select  cmslsnum,trgcount,deadtime,bitnum,prescale from trg where runnum=:runnum and bitname=:bitname;
select  cmslsnum,trgcount,deadtime,bitnum,prescale from trg where runnum=:runnum
select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnumber and d.algoname=:algoname and s.lumisummary_id=d.lumisummary_id order by s.startorbit,s.cmslsnum
select trghltmap.hltpathname,trghltmap.l1seed from cmsrunsummary,trghltmap where cmsrunsummary.runnum=:runnum and trghltmap.hltkey=cmsrunsummary.hltkey