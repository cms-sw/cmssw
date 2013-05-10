insert into tagruns (tagid,runnum) (select distinct 0, runnum from lumidata);
update tagruns set  lumidataid=(select max(data_id) from lumidata where lumidata.runnum=tagruns.runnum);
update tagruns set  trgdataid=(select max(data_id) from trgdata where trgdata.runnum=tagruns.runnum);
update tagruns set  hltdataid=(select max(data_id) from hltdata where hltdata.runnum=tagruns.runnum);
