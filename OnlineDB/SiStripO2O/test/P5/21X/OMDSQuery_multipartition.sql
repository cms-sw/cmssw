select
         r.runnumber as rNb
       , m.modedescription as modeT
       , p.partitionname as pName
       ,s2.fecversionmajorid||'.'||s2.fecversionminorid as new_fec
       , s2.fedversionmajorid||'.'||s2.fedversionminorid as new_fed
       , s2.connectionversionmajorid||'.'||s2.connectionversionminorid as new_conn
       , s2.dcuinfoversionmajorid||'.'||s2.dcuinfoversionminorid 
	, r.runmode
  from
    run r
  , partition p
  , modetype m
    , statehistory s2
  where
      m.runmode=r.runmode  
  and r.runmode !=21
--  and s2.connectionversionmajorid!=2
  and r.partitionid=p.partitionid
  and r.partitionid=s2.partitionid
  and r.statehistoryid=s2.statehistoryid
  and r.runnumber>49800
;
