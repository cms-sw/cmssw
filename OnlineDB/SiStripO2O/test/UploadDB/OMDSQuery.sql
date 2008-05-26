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
       from
         run r	
	, partition p
       , modetype m
       , analysis a
       , statehistory s
       , statehistory s2
       where
           m.runmode=r.runmode   
       and r.runmode=2
       and r.partitionid=p.partitionid
       and r.partitionid=s.partitionid
       and r.statehistoryid=s.statehistoryid
       and s.partitionid = s2.partitionid
       and r.runnumber = a.runnumber
       and r.partitionid = a.partitionid
       and s2.statehistoryid = (
           select min(s3.statehistoryid) -- we use the statehistoryid which is better than the date (no unique contraint on date) because of "order" option in the sequence generator
           from statehistory s3
           where
               s3.statehistoryid != s.statehistoryid
           and s3.partitionid = s.partitionid
         -- the two following constraints on state date tend to increase the reliability of the result by
         -- reducing the time window in which the partition state may be updated with an overlap
           and s3.statehistorydate > s.statehistorydate
           and s3.statehistorydate > a.analysisdate
         -- device parameter upload must also be perform then, versions must change
           and (
                   (     a.analysistype = 'FASTFEDCABLING' -- connection run
                     and s3.connectionversionmajorid > s.connectionversionmajorid -- connection version change
                   ) or (
                         a.analysistype != 'FASTFEDCABLING' and s3.connectionversionmajorid = s.connectionversionmajorid -- connection version doesn't change
                   )
           )
           -- 'APVLATENCY','CALIBRATION','FASTFEDCABLING','FINEDELAY','OPTOSCAN','PEDESTALS','TIMING','VPSPSCAN'
           and (
                   (     a.analysistype in ('APVLATENCY','CALIBRATION','FINEDELAY','OPTOSCAN','TIMING','VPSPSCAN')
                     and s3.fecversionmajorid > s.fecversionmajorid
                   ) or (
                         a.analysistype not in ('APVLATENCY','CALIBRATION','FINEDELAY','OPTOSCAN','TIMING','VPSPSCAN')
                     and s3.fecversionmajorid >= s.fecversionmajorid
                   )
           )
           and (
                   (     a.analysistype in ('FASTFEDCABLING','PEDESTALS','TIMING')
                     and s3.fedversionmajorid > s.fedversionmajorid
                   ) or (
                         a.analysistype not in ('FASTFEDCABLING','PEDESTALS','TIMING')
                     and s3.fedversionmajorid >= s.fedversionmajorid
                   )
           )
       )
--       	and r.runnumber= 38400
; 
