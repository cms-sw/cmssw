select
         r.runnumber as rNb
       , m.modedescription as modeT
       , p.partitionname as pName
       -- , a.analysistype
       -- , r.starttime as run_start
       -- , s.statehistorydate as state_date
       -- , s.fecversionmajorid||'.'||s.fecversionminorid as run_fec
       -- , s.fedversionmajorid||'.'||s.fedversionminorid as run_fed
       -- , s.connectionversionmajorid||'.'||s.connectionversionminorid as run_conn
       -- , s2.statehistorydate as newstate_date
       , s2.fecversionmajorid||'.'||s2.fecversionminorid as new_fec
       , s2.fedversionmajorid||'.'||s2.fedversionminorid as new_fed
       , s2.connectionversionmajorid||'.'||s2.connectionversionminorid as new_conn
       , s2.dcuinfoversionmajorid||'.'||s2.dcuinfoversionminorid 
	, r.runmode
  from
    run r
  , partition p
  , modetype m
  , statehistory s	
  , statehistory s2
  where
      m.runmode=r.runmode  
  and r.runmode !=21
  and s2.connectionversionmajorid!=2
  and r.partitionid=p.partitionid
  and r.partitionid=s.partitionid
  and r.statehistoryid=s.statehistoryid
  and s.partitionid = s2.partitionid
  and s2.statehistoryid = (
      select min(s3.statehistoryid)
      from statehistory s3
      where
          s3.statehistoryid > ( decode( s.statehistoryid, null, 0, s.statehistoryid) )
      and s3.partitionid = s.partitionid
  )
and ( 
	r.runmode!=2  or 
	exists (
  	 select x.analysisid
  	 from analysis x
  	 where x.runnumber = r.runnumber
   	and x.partitionid = r.partitionid
	) 
);
