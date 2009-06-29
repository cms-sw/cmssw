CREATE OR REPLACE FUNCTION OPEN_STATUS( LAST_WRITE_TIME IN DATE, s_created in number, s_injected in number) RETURN NUMBER

AS
	result_1   NUMBER;

BEGIN
	result_1 := 0;
	IF (ABS(time_diff(sysdate, LAST_WRITE_TIME)) > 180) THEN
		IF( (S_CREATED - S_INJECTED) > 0) THEN
		  result_1 := 2;
                END IF;
        END IF;
return result_1;
END;
/
CREATE OR REPLACE FUNCTION Safe0_CHECK ( s_safe0 in NUMBER, s_closed in NUMBER)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;
	IF ABS(s_closed - s_safe0) > 25 THEN
		status := 1;
	END IF;
	IF ABS(s_closed - s_safe0) > 100 THEN
		status :=2;
	END IF;
	RETURN status;			
END Safe0_CHECK;
/

CREATE OR REPLACE FUNCTION Safe99_CHECK ( lastWrite in DATE, lastTrans in DATE)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;
	IF (ABS(time_diff(sysdate, lastWrite)) < 180) THEN --Run is On
		IF (ABS(time_diff(sysdate, lastTrans)) > 300) THEN
			status := 2;
		END IF;
	END IF;
	--ELSE
	RETURN status;			
END Safe99_CHECK;
/

CREATE OR REPLACE FUNCTION TRANS_RATE_CHECK ( lastWrite in DATE, writeRate in Number, transRate in Number, startRun in DATE, totalSizeGB in Number)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;
	IF (writeRate = 0) THEN
		RETURN status;
	END IF;
	IF (ABS(time_diff(sysdate, lastWrite)) < 180) AND (ABS(time_diff(sysdate, startRun)) > 600) AND (totalSizeGB > 2) THEN
		IF ((writeRate - transRate) / writeRate) > .15 AND (writeRate - transRate) > 10 THEN 
			status := 1;
		END IF;
		IF ((writeRate - transRate) / writeRate) > .30 AND (writeRate - transRate) > 50 THEN
			IF (writeRate < 1000) THEN
				status := 2;
			ELSE
				status := 1;
			END IF;
		END IF;
	ELSE --Run is off	
		IF ((writeRate - transRate) / writeRate) > .15 AND (writeRate - transRate) > 10 THEN 
			status := 1;
		END IF;
		IF ((writeRate - transRate) / writeRate) > .30 AND (writeRate - transRate) > 50 THEN
			IF (writeRate < 1000) THEN
				status := 2;
			ELSE
				status := 1;
			END IF;
		END IF;	
	END IF;
	RETURN status;			
END TRANS_RATE_CHECK;
/

CREATE or REPLACE FUNCTION DELETED_CHECK ( Start_time in DATE, s_deleted in number, s_checked in number, LastTrans in DATE) 
   RETURN NUMBER AS
   result_1    number;
BEGIN
    result_1 := 0;
    IF s_checked = 0 THEN
	return result_1; 
    END IF;
    IF ( (time_diff(sysdate, LastTrans)) < 9000) THEN
        IF ( (s_checked  - s_deleted )  > (7200 * ( s_checked / time_diff( sysdate, Start_time )))) THEN
        result_1 := 1;
        END IF;
    ELSE
        IF ( ABS(S_checked - s_deleted) > 0 ) THEN
        result_1 := 2;
        END IF;
    END IF;

return result_1;
END DELETED_CHECK;
/

create or replace view V_SM_SUMMARY
AS SELECT  "RUN_NUMBER", 
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
	   "N_SAFE0", 
	   "N_SAFE99",
           "MAX_SAFE99_INSTANCE",
	   "MIN_SAFE99_INSTANCE", 
	   "N_DELETED",
	   "N_REPACKED",
	   "HLTKEY",
	   "SETUP_STATUS",
	   "N_OPEN_STATUS",
	   "WRITE_STATUS",
	   "TRANS_STATUS",
	   "SAFE0_STATUS",
	   "SAFE99_STATUS",
	   "DELETED_STATUS" 
FROM ( SELECT TO_CHAR ( RUN ) AS RUN_NUMBER,
	      TO_CHAR ( MIN( startwt ), 'dd.mm.yyyy hh24:mi:ss' ) AS START_TIME,
	      TO_CHAR ( MAX( lut ), 'dd.mm.yyyy hh24:mi:ss' ) AS UPDATE_TIME,
	      TO_CHAR ( MAX( setup ) ) AS SETUPLABEL,
	      TO_CHAR ( MAX( app_v ) ) AS APP_VERSION,
	      TO_CHAR ( MAX( inst ) + 1 ) AS M_INSTANCE,
	      TO_CHAR ( ROUND (SUM(NVL(fsize,0))/1073741824, 2) ) AS TOTAL_SIZE,
	      TO_CHAR ( SUM(NVL(created, 0)) ) AS NFILES,
	     (CASE SUM(NVL(injected,0))
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((SUM(NVL(size2D,0)) / 1048576) / (GREATEST(time_diff( MAX(stopwt), MIN(startwt)),1)), 2))
	      END) AS RATE2D_AVG,  
	     (CASE SUM(NVL(copied,0))
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((SUM(NVL(size2T0,0)) / 1048576) / (GREATEST(time_diff( MAX(stoptt), MIN(starttt)),1)), 2))
	      END) AS RATE2T_AVG,	      
	      TO_CHAR ( SUM(NVL(events,0)) ) AS NEVTS, 
	      TO_CHAR ( SUM(NVL(created,0) ) - SUM( NVL(Injected,0) ) ) AS N_OPEN, 
	      TO_CHAR ( SUM(NVL(injected, 0) ) ) AS N_CLOSED, 
	      TO_CHAR ( SUM(NVL(new, 0) ) ) AS N_SAFE0, 
	      TO_CHAR ( SUM(NVL(checked, 0) ) ) AS N_SAFE99,
	      TO_CHAR ( MAX(NVL(safe99,0)) ) AS MAX_SAFE99_INSTANCE,
	      TO_CHAR ( MIN(NVL(safe99,0)) ) AS MIN_SAFE99_INSTANCE,
	      TO_CHAR ( SUM(NVL(deleted, 0) ) ) AS N_DELETED, 
	      TO_CHAR ( SUM(NVL(repacked, 0) ) ) AS N_REPACKED,
	      TO_CHAR ( MAX(HLT) ) AS HLTKEY,
             (CASE
                WHEN MAX(setup) LIKE '%TransferTest%' THEN TO_CHAR(2)
                ELSE TO_CHAR(0)
              END) AS SETUP_STATUS,
	      TO_CHAR ( OPEN_STATUS(MAX(stopwt), SUM(NVL(CREATED,0)), SUM(NVL(INJECTED,0))) ) AS N_OPEN_STATUS,
	     (CASE 
		WHEN (CASE SUM(NVL(injected, 0))
		        WHEN 0 THEN 0
		        ELSE ROUND ((SUM(NVL(size2D,0)) / 1048576) / (GREATEST(time_diff( MAX(stopwt), MIN(startwt)),1)), 2)
	              END) < 2000 THEN TO_CHAR(0)
	        ELSE TO_CHAR(1)
	      END) AS WRITE_STATUS,
	      TO_CHAR ( TRANS_RATE_CHECK(MAX(stopwt),
		                        (CASE SUM(NVL(injected,0))
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((SUM(NVL(size2D,0)) / 1048576) / (GREATEST(time_diff( MAX(stopwt), MIN(startwt)),1)), 2)
	                                 END),
		                        (CASE SUM(NVL(copied,0))
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((SUM(NVL(size2T0,0)) / 1048576) / (GREATEST(time_diff( MAX(stoptt), MIN(starttt)),1)), 2)
	                                 END),
					 MIN(startwt),
					 ROUND (SUM(NVL(fsize,0))/1073741824, 2))) AS TRANS_STATUS,
	      TO_CHAR ( SAFE0_CHECK(SUM(NVL(NEW,0)), SUM(NVL(injected,0))) ) AS SAFE0_STATUS,
	      TO_CHAR ( SAFE99_CHECK(MAX(stopwt), MAX(stoptt)) ) AS SAFE99_STATUS,
	      TO_CHAR ( DELETED_CHECK(MIN(startwt), SUM(NVL(Deleted, 0)), SUM(NVL(Checked, 0)), MAX(stoptt)) ) AS DELETED_STATUS
FROM (SELECT    a.RUNNUMBER as run, 
		a.STREAM as str, 
		a.SETUPLABEL as setup, 
		a.APP_VERSION as app_v, 
		a.S_LUMISECTION as lumi, 
		a.S_FILESIZE as fsize, 
		a.S_FILESIZE2D as size2D, 
		a.S_FILESIZE2T0 as size2T0, 
		a.S_NEVENTS as events, 
		a.S_CREATED as created, 
		a.S_INJECTED as injected, 
		a.S_NEW as new, 
		a.S_COPIED as copied, 
		a.S_CHECKED as checked,  
		a.S_INSERTED as inserted, 
		a.S_REPACKED as repacked, 
		a.S_DELETED as deleted, 
		a.M_INSTANCE as inst, 
		a.START_WRITE_TIME as startwt, 
		a.STOP_WRITE_TIME as stopwt, 
		a.START_TRANS_TIME as starttt, 
		a.STOP_TRANS_TIME as stoptt, 
		a.START_REPACK_TIME as startrt, 
		a.STOP_REPACK_TIME as stoprt, 
		a.HLTKEY as hlt, 
		a.LAST_UPDATE_TIME as lut, 
		b.RUNNUMBER, 
		b.INSTANCE, 
		b.N_CHECKED as safe99, 
		DENSE_RANK() OVER (ORDER BY a.RUNNUMBER DESC NULLS LAST) as runRank 
FROM SM_SUMMARY a FULL OUTER JOIN SM_INSTANCES b ON a.RUNNUMBER=b.RUNNUMBER)
WHERE runRank <= 120
GROUP BY run
)
ORDER BY 1 DESC;


grant select on V_SM_SUMMARY to public;
