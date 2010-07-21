CREATE OR REPLACE FUNCTION NOTFILES_CHECK2( run in number, LAST_WRITE_TIME IN DATE, s_notfiles in number, n_nodes in number) 
    RETURN NUMBER AS
    result_1   NUMBER;
    nstream    NUMBER;
BEGIN
	result_1 := 0;

        --need to get info from SUMMARY TABLE:
        SELECT COUNT(CMS_STOMGR.SM_SUMMARY.STREAM) INTO nstream from CMS_STOMGR.SM_SUMMARY WHERE RUNNUMBER = run;

        IF( (s_notfiles > 0) AND (1260 < ABS(time_diff(sysdate, LAST_WRITE_TIME)) ) ) THEN   
	   IF ( ABS(time_diff(sysdate, LAST_WRITE_TIME)) < 2700 ) THEN
                   result_1 := 1;
           ELSE
	           result_1 := 2;
           END IF;
        END IF;

--no matter what, if count is too large go RED:
        IF( (s_notfiles - 0.5 * nstream * n_nodes) > 0) THEN  
	   result_1 := 2;
        END IF;


return result_1;
END NOTFILES_CHECK2;
/



CREATE OR REPLACE FUNCTION OPEN_STATUS( LAST_WRITE_TIME IN DATE, s_created in number, s_injected in number) 
    RETURN NUMBER AS
    result_1   NUMBER;
BEGIN
	result_1 := 0;
	--If last write was more than 7 min ago (run is assumed over), turns magenta if open files not zero
	IF (ABS(time_diff(sysdate, LAST_WRITE_TIME)) > 360) THEN
		IF( (S_CREATED - S_INJECTED) > 0) THEN
		  result_1 := 1;
                END IF;
        END IF;
return result_1;
END;
/

CREATE OR REPLACE FUNCTION INJECTED_CHECK ( s_safe0 in NUMBER, s_closed in NUMBER,  s_deleted in number, lastWrite in DATE)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN

        status := 0;
    --if there is any difference, display with blue color, override later if more severe conditions
    IF ( s_closed != s_safe0 ) THEN
         status := 3;
    END IF;

	--if last write was less than 5 min ago (running)
	IF (ABS(time_diff(sysdate, lastWrite)) < 300) THEN
		--magenta if more than 350 files waiting to be injected (sort of ~1 min lag)
		IF ABS(s_closed - s_safe0) > 350 THEN
			status := 1;
		END IF;
		--red if more than 1700 files waiting to be injected (sort of ~5 min lag)
		IF ABS(s_closed - s_safe0) > 1700 THEN
			status :=2;
		END IF;
	--if not running
	ELSE
		--red if any files waiting to be injected
		IF (s_closed - s_safe0 > 0) THEN
			status := 2;
                        --special treatment cuz of DontNotifyT0 files: IF CLOSED=DELETED red->magenta  [?? valid??]
	   	        IF ( s_closed - s_deleted = 0 ) THEN
			    status := 1;
                        END IF;
                END IF;
	END IF;
   

	RETURN status;			
END INJECTED_CHECK;
/

CREATE OR REPLACE FUNCTION TRANSFERRED_CHECK ( lastWrite in DATE, lastTrans in DATE, s_new in number, s_copied in number)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;

    --if there is any difference, display with blue color, override later if more severe conditions
    IF ( s_new != s_copied ) THEN
         status := 3;
    END IF;


	--if last write was less than 3 min ago
	IF (ABS(time_diff(sysdate, lastWrite)) < 180) THEN --Currently happening
		--turn red if it's been more than 6 min since last transfer
		IF (ABS(time_diff(sysdate, lastTrans)) > 360) THEN
			status := 2;
		END IF;
	--last write was more than 3 min ago
	ELSE
		--turn red if it's been 6 min since last transfer and all files haven't been transferred
		IF ( (ABS(time_diff(sysdate, lastTrans)) > 360) AND (s_new - s_copied > 0) ) THEN
                        status := 2;
		END IF;
        END IF;

	RETURN status;			
END TRANSFERRED_CHECK;
/

CREATE OR REPLACE FUNCTION CHECKED_CHECK ( lastWrite in DATE, lastTrans in DATE, s_checked in number, s_copied in number)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;

    --if there is any difference, display with blue color, override later if more severe conditions
    IF ( s_copied != s_checked ) THEN
         status := 3;
    END IF;

	--if last write less than 3 min ago currently no check
	IF (ABS(time_diff(sysdate, lastWrite)) > 180) THEN --Run is Off
--		status:= 0;

	--if last check more than 10 min ago and not all transferred files checked turn magenta
		IF ( (ABS(time_diff(sysdate, lastTrans)) > 600) AND (s_copied - s_checked > 0) ) THEN
                        status := 1;
		END IF;
        END IF;

	RETURN status;			
END CHECKED_CHECK;
/

CREATE OR REPLACE FUNCTION TRANS_RATE_CHECK ( lastWrite in DATE, writeRate in Number, transRate in Number, startRun in DATE, totalSizeGB in Number, numInst in Number)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;
	
	--This checks if transfers aren't happening at all
	IF (time_diff(lastWrite,startRun) > 360) AND (writeRate > 0) AND (transRate = 0) THEN
		status := 2;
                RETURN status;
	END IF;
	
	--Otherwise only check if size greater than certain lower limit
	IF (totalSizeGB > 50) THEN
		--if write rate is low, turn magenta if transfer rate is less than 10% of write rate
		IF (writeRate < 20) THEN
			IF (transRate < .1 * writeRate) THEN
				status := 1;  --Used to be red (2)
			END IF;
		--high write rate
		ELSE
			--if transfer rate is not too high, flags if transfer rate is relatively too low (currently only magenta)
			IF (transRate + 5 < 0.75 * writeRate) AND (transRate < 900 * (numInst / 16)) THEN
				status := 1;
			END IF;
			IF (transRate < .5 * writeRate) AND (transRate < 800 * (numInst / 16)) THEN
				status := 1;  --Used to be red (2)
			END IF;
		END IF;
	END IF;

	RETURN status;			
END TRANS_RATE_CHECK;
/

CREATE or REPLACE FUNCTION DELETED_CHECK (n_hosts in number, Start_time in DATE, s_deleted in number, s_checked in number, s_closed in number, LastTrans in DATE) 
   RETURN NUMBER AS
   result_1      number;
   --delete      frequency
   dfreq         number;
   toffset       number;
BEGIN
   dfreq  := 20*60;
  


    result_1 := 0;
    --no test: if no files "checked" that can be deleted
    IF (  s_checked = s_deleted ) THEN
	return result_1; 
    END IF;


    --if there is any difference, display with blue color, override later if more severe conditions
    IF ( s_checked != s_deleted ) THEN
       result_1 := 3;
    END IF;


    --if n_hosts=1 ASSUME we are talking MiniDaq!? DELETES done only ONCE per day (15:07 hrs)
    IF( n_hosts = 1) THEN
       --key: 60 sec/min * ( 60 min/hr * (15:00 minidaq delete time + 2 hr delete delay) + 20 min grace)
       IF (   time_diff(TRUNC(sysdate,'DD'), LastTrans) - 60*(60*(15+2) + 20) > 0        ) THEN
         result_1 := 1;
         IF ( time_diff(TRUNC(sysdate,'DD'), LastTrans) - 60*(60*(15+2) + 20) > 60*60*25 ) THEN
           result_1 := 2;
         END IF; 
       END IF; 
       return result_1; 
    ELSE

--     toffset :=  time_diff( LastTrans, Start_time ) + dfreq +  2*dfreq*8*(1 - mod(n_hosts,8)/8);
--     toffset :=  time_diff( LastTrans, Start_time ) + dfreq +  2*dfreq*8;
--     nominal delete cycle:
       toffset :=  2*dfreq*(9+1);



       --if run is ~ongoing (transfs in last 5 min):
       IF ( time_diff(sysdate, LastTrans) < 5*60) THEN


         IF (   s_deleted/s_checked < 1.0 - (toffset -  3*dfreq)/time_diff( LastTrans, Start_time ) ) THEN
           result_1 := 1;
           IF ( s_deleted/s_checked < 1.0 - (toffset +  1*dfreq)/time_diff( LastTrans, Start_time ) ) THEN
              result_1 := 2;
           END IF;
         END IF;
         return result_1; 
      END IF;


     --after run is over (no transf for 5 min):
     IF (   s_deleted/s_checked < 1.0 - GREATEST(toffset +  5*dfreq - time_diff(sysdate, LastTrans ), 0.0)/time_diff( LastTrans, Start_time ) ) THEN
       result_1 := 1;
       IF ( s_deleted/s_checked < 1.0 - GREATEST(toffset +  7*dfreq - time_diff(sysdate, LastTrans ), 0.0)/time_diff( LastTrans, Start_time ) ) THEN
          result_1 := 2;
       END IF;
     END IF;


  END IF;


return result_1;
END DELETED_CHECK;
/


--Provides per run summary information (one row per run)
create or replace view V_SM_SUMMARY
AS SELECT  
           "RUN_NUMBER", 
 	   "START_TIME",
	   "UPDATE_TIME",
           "SETUPLABEL",
	   "APP_VERSION",
	   "M_INSTANCE", 
	   "TOTAL_SIZE",
	   "NFILES", 
	   "NEVTS",
	   "RATE2D_AVG",
	   "RATE2T_AVG", 
	   "N_OPEN", 
	   "N_CLOSED",
	   "N_INJECTED",
           "N_TRANSFERRED", 
	   "N_CHECKED", 
	   "N_DELETED",
	   "N_REPACKED",
	   "HLTKEY",
	   "SETUP_STATUS",
           "NOTFILES_STATUS",
	   "N_OPEN_STATUS",
	   "WRITE_STATUS",
	   "TRANS_STATUS",
	   "INJECTED_STATUS",
           "CHECKED_STATUS",
	   "TRANSFERRED_STATUS",
	   "DELETED_STATUS",
	   "NOTFILES",
           "RANK" 
FROM (  SELECT  TO_CHAR( RUNNUMBER )          AS RUN_NUMBER,
                TO_CHAR( COUNT( INSTANCE ) )  AS NOT_INSTANCES, 
	        TO_CHAR( SUM( N_UNACCOUNT ) ) AS NOTFILES,
                TO_CHAR( NOTFILES_CHECK2(RUNNUMBER, MAX(LAST_WRITE_TIME), SUM(NVL(N_UNACCOUNT,0)), COUNT(INSTANCE) ) ) AS NOTFILES_STATUS,
                TO_CHAR( MAX(runRank) )       AS RANK
                FROM ( SELECT RUNNUMBER, INSTANCE, N_UNACCOUNT, HOSTNAME,LAST_WRITE_TIME,  
                           DENSE_RANK() OVER (ORDER BY SM_INSTANCES.RUNNUMBER DESC NULLS LAST)
                           runRank FROM SM_INSTANCES)
                    WHERE runRank <= 120
                    GROUP BY RUNNUMBER
            --        ORDER BY RUNNUMBER DESC  --
     ),
     ( SELECT TO_CHAR ( RUNNUMBER )                                      AS RUN_NUMBER2,
	      TO_CHAR ( MIN(START_WRITE_TIME), 'dd.mm.yyyy hh24:mi:ss' ) AS START_TIME,
	      TO_CHAR ( MAX(LAST_UPDATE_TIME), 'dd.mm hh24:mi:ss' )      AS UPDATE_TIME,
	      TO_CHAR ( MAX(setupLabel) )                                AS SETUPLABEL,
	      TO_CHAR ( MAX(APP_VERSION) )                               AS APP_VERSION,
	      TO_CHAR ( NVL(MAX(N_INSTANCE), MAX(M_INSTANCE) + 1) )      AS M_INSTANCE,
	     (CASE SUM(NVL(s_filesize, 0))
                WHEN 0 THEN TO_CHAR('NA')
                ELSE TO_CHAR ( ROUND (SUM(NVL(s_filesize,0))/1073741824, 2) )
              END) AS TOTAL_SIZE,
	      TO_CHAR ( SUM(NVL(s_Created, 0)) )                         AS NFILES,
	     (CASE SUM(NVL(s_injected,0))
		WHEN 0 THEN TO_CHAR('NA')
		ELSE TO_CHAR ( ROUND ((SUM(NVL(s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_WRITE_TIME), MIN(START_WRITE_TIME)),1)), 2))
	      END)                                                       AS RATE2D_AVG,  
	     (CASE SUM(NVL(s_copied,0))
		WHEN 0 THEN TO_CHAR('NA')
		ELSE TO_CHAR ( ROUND ((SUM(NVL(s_filesize2T0,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_TRANS_TIME), MIN(START_TRANS_TIME)),1)), 2))
	      END)                                                       AS RATE2T_AVG,	      
	     (CASE SUM(NVL(s_NEvents,0))
                WHEN 0 THEN TO_CHAR('NA')
                ELSE TO_CHAR ( SUM(NVL(s_NEvents,0)) )
              END)                                                       AS NEVTS,
	      -- N_OPEN = N_FILES - N_CLOSED
	      TO_CHAR ( SUM(NVL(s_Created,0) ) - SUM( NVL(s_Injected,0))) AS N_OPEN, 
	      TO_CHAR ( SUM(NVL(s_Injected, 0) ) )                        AS N_CLOSED, 
	      TO_CHAR ( SUM(NVL(s_New, 0) ) )                             AS N_INJECTED,
              TO_CHAR ( SUM(NVL(s_Copied, 0) ) )                          AS N_TRANSFERRED, 
	      TO_CHAR ( SUM(NVL(s_Checked, 0) ) )                         AS N_CHECKED,
	      TO_CHAR ( SUM(NVL(s_Deleted, 0) ) )                         AS N_DELETED, 
	      TO_CHAR ( SUM(NVL(s_Repacked, 0) ) )                        AS N_REPACKED,
	      --This will turn red if the hltkey contains the phrase "TransferTest"
	      TO_CHAR ( MAX(HLTKEY) )                                     AS HLTKEY,
             (CASE
                WHEN MAX(setupLabel) LIKE '%TransferTest%' THEN TO_CHAR(2)
                ELSE TO_CHAR(0)
              END)                                                        AS SETUP_STATUS,
	      --This is the check for the write rate - it turns magenta if the rate is more than 2000 mbs scaled by active instances
	      TO_CHAR ( OPEN_STATUS(NVL(MAX(STOP_WRITE_TIME), MAX(LAST_UPDATE_TIME)), SUM(NVL(S_CREATED,0)), SUM(NVL(S_INJECTED,0))) ) AS N_OPEN_STATUS,
	     (CASE 
		WHEN (CASE SUM(NVL(s_injected, 0))
		        WHEN 0 THEN 0
		        ELSE ROUND ((SUM(NVL(s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_WRITE_TIME), MIN(START_WRITE_TIME)),1)), 2)
	              END) < ((MAX(M_INSTANCE) + 1)/ 16) * 2000 THEN TO_CHAR(0)
	        ELSE TO_CHAR(1)
	      END)                                                        AS WRITE_STATUS,
	      TO_CHAR ( TRANS_RATE_CHECK(MAX(STOP_WRITE_TIME),
		                        (CASE SUM(NVL(s_injected,0))
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((SUM(NVL(s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_WRITE_TIME), MIN(START_WRITE_TIME)),1)), 2)
	                                 END),
		                        (CASE SUM(NVL(s_copied,0))
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((SUM(NVL(s_filesize2T0,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_TRANS_TIME), MIN(START_TRANS_TIME)),1)), 2)
	                                 END),
					 MIN(START_WRITE_TIME),
					 ROUND (SUM(NVL(s_filesize,0))/1073741824, 2),
                                         MAX(M_INSTANCE) + 1))             AS TRANS_STATUS,
	      TO_CHAR ( INJECTED_CHECK(SUM(NVL(s_NEW,0)), SUM(NVL(s_injected,0)),  SUM(NVL(s_Deleted, 0)), MAX(STOP_WRITE_TIME) ) ) AS INJECTED_STATUS,
	      TO_CHAR ( TRANSFERRED_CHECK(MAX(STOP_WRITE_TIME), MAX(STOP_TRANS_TIME), SUM(NVL(s_NEW,0)), SUM(NVL(s_Copied,0)) ) )   AS TRANSFERRED_STATUS,
	      TO_CHAR ( CHECKED_CHECK(MAX(STOP_WRITE_TIME), MAX(STOP_TRANS_TIME), SUM(NVL(s_CHECKED,0)), SUM(NVL(s_COPIED,0)) ) )   AS CHECKED_STATUS,
	      TO_CHAR ( DELETED_CHECK( COUNT(DISTINCT N_INSTANCE), MIN(START_WRITE_TIME), SUM(NVL(s_Deleted, 0)), SUM(NVL(s_Checked, 0)), SUM(NVL(s_injected,0)), MAX(STOP_TRANS_TIME)) ) AS DELETED_STATUS
         FROM (  SELECT  RUNNUMBER, STREAM, SETUPLABEL, APP_VERSION, S_LUMISECTION, 
                 S_FILESIZE, S_FILESIZE2D, S_FILESIZE2T0, S_NEVENTS, S_CREATED, S_INJECTED, 
                 S_NEW,S_COPIED,S_CHECKED,S_INSERTED,S_REPACKED,S_DELETED, N_INSTANCE,M_INSTANCE,
                 START_WRITE_TIME,STOP_WRITE_TIME,START_TRANS_TIME,STOP_TRANS_TIME,START_REPACK_TIME,
                 STOP_REPACK_TIME, HLTKEY,LAST_UPDATE_TIME
                 FROM SM_SUMMARY )
         GROUP BY RUNNUMBER
--         ORDER BY RUNNUMBER DESC
   ) WHERE RUN_NUMBER=RUN_NUMBER2(+)
                    ORDER BY RUN_NUMBER DESC ;
 
grant select on V_SM_SUMMARY to public;
