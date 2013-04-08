insert into pixeltagruns (tagid,runnum) (select distinct 0, runnum from pixellumidata);
update pixeltagruns set  lumidataid=(select max(data_id) from pixellumidata where pixellumidata.runnum=pixeltagruns.runnum);
update pixeltagruns set  trgdataid=(select max(data_id) from trgdata where trgdata.runnum=pixeltagruns.runnum);
update pixeltagruns set  hltdataid=(select max(data_id) from hltdata where hltdata.runnum=pixeltagruns.runnum);
