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
	      TO_CHAR ( MIN( START_WRITE_TIME ), 'dd.mm.yyyy hh24:mi:ss' ) AS START_TIME,
	      TO_CHAR ( MAX(a.LAST_UPDATE_TIME), 'dd.mm.yyyy hh24:mi:ss' ) AS UPDATE_TIME,
	      TO_CHAR ( MAX(a.setupLabel) ) AS SETUPLABEL,
	      TO_CHAR ( MAX(a.APP_VERSION) ) AS APP_VERSION,
	      TO_CHAR ( MAX(a.M_INSTANCE) + 1 ) AS M_INSTANCE,
	      TO_CHAR ( ROUND (SUM(NVL(a.s_filesize,0))/1073741824, 2) ) AS TOTAL_SIZE,
	      TO_CHAR ( SUM(NVL(a.s_Created, 0)) ) AS NFILES,
	     (CASE SUM(NVL(a.s_injected,0))
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((SUM(NVL(a.s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(a.STOP_WRITE_TIME), MIN(a.START_WRITE_TIME)),1)), 2))
	      END) AS RATE2D_AVG,  
	     (CASE SUM(NVL(a.s_copied,0))
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((SUM(NVL(a.s_filesize2T0,0)) / 1048576) / (GREATEST(time_diff( MAX(stop), MIN(a.START_TRANS_TIME)),1)), 2))
	      END) AS RATE2T_AVG,	      
	      TO_CHAR ( SUM(NVL(a.s_NEvents,0)) ) AS NEVTS, 
	      TO_CHAR ( SUM(NVL(a.s_Created,0) ) - SUM( NVL(a.s_Injected,0) ) ) AS N_OPEN, 
	      TO_CHAR ( SUM(NVL(a.s_Injected, 0) ) ) AS N_CLOSED, 
	      TO_CHAR ( SUM(NVL(a.s_New, 0) ) ) AS N_SAFE0, 
	      TO_CHAR ( SUM(NVL(a.s_Checked, 0) ) ) AS N_SAFE99,
	      TO_CHAR ( SUM(NVL(a.s_Deleted, 0) ) ) AS N_DELETED, 
	      TO_CHAR ( SUM(NVL(a.s_Repacked, 0) ) ) AS N_REPACKED,
	      TO_CHAR ( MAX(a.HLTKEY) ) AS HLTKEY,
             (CASE
                WHEN MAX(a.setupLabel) LIKE '%TransferTest%' THEN TO_CHAR(2)
                ELSE TO_CHAR(0)
              END) AS SETUP_STATUS,
	      TO_CHAR ( OPEN_STATUS(MAX(a.STOP_WRITE_TIME), SUM(NVL(a.S_CREATED,0)), SUM(NVL(a.S_INJECTED,0))) ) AS N_OPEN_STATUS,
	     (CASE 
		WHEN (CASE SUM(NVL(a.s_injected, 0))
		        WHEN 0 THEN 0
		        ELSE ROUND ((SUM(NVL(a.s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(a.STOP_WRITE_TIME), MIN(a.START_WRITE_TIME)),1)), 2)
	              END) < 2000 THEN TO_CHAR(0)
	        ELSE TO_CHAR(1)
	      END) AS WRITE_STATUS,
	      TO_CHAR ( TRANS_RATE_CHECK(MAX(a.STOP_WRITE_TIME),
		                        (CASE SUM(NVL(a.s_injected,0))
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((SUM(NVL(a.s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(a.STOP_WRITE_TIME), MIN(a.START_WRITE_TIME)),1)), 2)
	                                 END),
		                        (CASE SUM(NVL(a.s_copied,0))
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((SUM(NVL(a.s_filesize2T0,0)) / 1048576) / (GREATEST(time_diff( MAX(stop), MIN(a.START_TRANS_TIME)),1)), 2)
	                                 END),
					 MIN(a.START_WRITE_TIME),
					 ROUND (SUM(NVL(a.s_filesize,0))/1073741824, 2))) AS TRANS_STATUS,
	      TO_CHAR ( SAFE0_CHECK(SUM(NVL(a.s_NEW,0)), SUM(NVL(a.s_injected,0))) ) AS SAFE0_STATUS,
	      TO_CHAR ( SAFE99_CHECK(MAX(a.STOP_WRITE_TIME), MAX(stop)) ) AS SAFE99_STATUS,
	      TO_CHAR ( DELETED_CHECK(MIN(a.START_WRITE_TIME), SUM(NVL(a.s_Deleted, 0)), SUM(NVL(a.s_Checked, 0)), MAX(stop)) ) AS DELETED_STATUS
FROM (SELECT a.RUNNUMBER, a.STREAM, a.SETUPLABEL, a.APP_VERSION, a.S_LUMISECTION, a.S_FILESIZE, a.S_FILESIZE2D, a.S_FILESIZE2T0, a.S_NEVENTS, a.S_CREATED, a.S_INJECTED, a.S_NEW, a.S_COPIED, a.S_CHECKED, a.S_INSERTED, a.S_REPACKED, a.S_DELETED, a.M_INSTANCE, a.START_WRITE_TIME, a.STOP_WRITE_TIME, a.START_TRANS_TIME, a.STOP_TRANS_TIME as stop, a.START_REPACK_TIME, a.STOP_REPACK_TIME, a.HLTKEY, a.LAST_UPDATE_TIME, b.RUNNUMBER, b.INSTANCE, b.N_SAFE99, DENSE_RANK() OVER (ORDER BY a.RUNNUMBER DESC NULLS LAST) as runRank 
FROM SM_SUMMARY a JOIN SM_INSTANCES b ON a.RUNNUMBER=b.RUNNUMBER
GROUP BY a.RUNNUMBER)
WHERE runRank <= 120)
ORDER BY 1 DESC;


grant select on V_SM_SUMMARY to public;
