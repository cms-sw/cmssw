CREATE OR REPLACE FUNCTION OPEN_STATUS( LAST_WRITE_TIME IN DATE, s_created in number, s_injected in number) RETURN NUMBER

AS
	result_1   NUMBER;

BEGIN
	result_1 := 0;
	IF (ABS(time_diff(sysdate, LAST_WRITE_TIME)) > 180) THEN
		IF( (S_CREATED - S_INJECTED) > 0) THEN
		  result_1 := 1;
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

CREATE OR REPLACE FUNCTION TRANS_RATE_CHECK ( lastWrite in DATE, writeRate in Number, transRate in Number, startRun in DATE, totalSizeGB in Number, numInst in Number)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;
	IF (writeRate < 50 OR totalSizeGB < 100) THEN
		RETURN status;
	END IF;
	IF (ABS(time_diff(sysdate, lastWrite)) < 180) AND (ABS(time_diff(sysdate, startRun)) > 600)  THEN
		IF ((writeRate - transRate) / writeRate) > .30 THEN 
			status := 1;
		END IF;
		IF ((writeRate - transRate) / writeRate) > .50 AND (writeRate - transRate) > 50 THEN
			IF (writeRate < (numInst / 16) *1000) THEN
				status := 2;
			ELSE
				status := 1;
			END IF;
		END IF;
	ELSE --Run is off	
		IF ((writeRate - transRate) / writeRate) > .30  THEN 
			status := 1;
		END IF;
		IF ((writeRate - transRate) / writeRate) > .50 AND (writeRate - transRate) > 50 THEN
			IF (writeRate < (numInst / 16) * 1000) THEN
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
FROM ( SELECT TO_CHAR ( RUNNUMBER ) AS RUN_NUMBER,
	      TO_CHAR ( MIN(START_WRITE_TIME), 'dd.mm.yyyy hh24:mi:ss' ) AS START_TIME,
	      TO_CHAR ( MAX(LAST_UPDATE_TIME), 'dd.mm.yyyy hh24:mi:ss' ) AS UPDATE_TIME,
	      TO_CHAR ( MAX(setupLabel) ) AS SETUPLABEL,
	      TO_CHAR ( MAX(APP_VERSION) ) AS APP_VERSION,
	      TO_CHAR ( MAX(M_INSTANCE) + 1 ) AS M_INSTANCE,
	      TO_CHAR ( ROUND (SUM(NVL(s_filesize,0))/1073741824, 2) ) AS TOTAL_SIZE,
	      TO_CHAR ( SUM(NVL(s_Created, 0)) ) AS NFILES,
	     (CASE SUM(NVL(s_injected,0))
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((SUM(NVL(s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_WRITE_TIME), MIN(START_WRITE_TIME)),1)), 2))
	      END) AS RATE2D_AVG,  
	     (CASE SUM(NVL(s_copied,0))
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((SUM(NVL(s_filesize2T0,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_TRANS_TIME), MIN(START_TRANS_TIME)),1)), 2))
	      END) AS RATE2T_AVG,	      
	      TO_CHAR ( SUM(NVL(s_NEvents,0)) ) AS NEVTS, 
	      TO_CHAR ( SUM(NVL(s_Created,0) ) - SUM( NVL(s_Injected,0) ) ) AS N_OPEN, 
	      TO_CHAR ( SUM(NVL(s_Injected, 0) ) ) AS N_CLOSED, 
	      TO_CHAR ( SUM(NVL(s_New, 0) ) ) AS N_SAFE0, 
	      TO_CHAR ( SUM(NVL(s_Checked, 0) ) ) AS N_SAFE99,
	      TO_CHAR ( SUM(NVL(s_Deleted, 0) ) ) AS N_DELETED, 
	      TO_CHAR ( SUM(NVL(s_Repacked, 0) ) ) AS N_REPACKED,
	      TO_CHAR ( MAX(HLTKEY) ) AS HLTKEY,
             (CASE
                WHEN MAX(setupLabel) LIKE '%TransferTest%' THEN TO_CHAR(2)
                ELSE TO_CHAR(0)
              END) AS SETUP_STATUS,
	      TO_CHAR ( OPEN_STATUS(MAX(STOP_WRITE_TIME), SUM(NVL(S_CREATED,0)), SUM(NVL(S_INJECTED,0))) ) AS N_OPEN_STATUS,
	     (CASE 
		WHEN (CASE SUM(NVL(s_injected, 0))
		        WHEN 0 THEN 0
		        ELSE ROUND ((SUM(NVL(s_filesize2D,0)) / 1048576) / (GREATEST(time_diff( MAX(STOP_WRITE_TIME), MIN(START_WRITE_TIME)),1)), 2)
	              END) < ((MAX(M_INSTANCE) + 1)/ 16) * 2000 THEN TO_CHAR(0)
	        ELSE TO_CHAR(1)
	      END) AS WRITE_STATUS,
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
                                         MAX(M_INSTANCE) + 1)) AS TRANS_STATUS,
	      TO_CHAR ( SAFE0_CHECK(SUM(NVL(s_NEW,0)), SUM(NVL(s_injected,0))) ) AS SAFE0_STATUS,
	      TO_CHAR ( SAFE99_CHECK(MAX(STOP_WRITE_TIME), MAX(STOP_TRANS_TIME)) ) AS SAFE99_STATUS,
	      TO_CHAR ( DELETED_CHECK(MIN(START_WRITE_TIME), SUM(NVL(s_Deleted, 0)), SUM(NVL(s_Checked, 0)), MAX(STOP_TRANS_TIME)) ) AS DELETED_STATUS
FROM (SELECT RUNNUMBER, STREAM, SETUPLABEL, APP_VERSION, S_LUMISECTION, S_FILESIZE, S_FILESIZE2D, S_FILESIZE2T0, S_NEVENTS, S_CREATED, S_INJECTED, S_NEW, S_COPIED, S_CHECKED, S_INSERTED, S_REPACKED, S_DELETED, M_INSTANCE, START_WRITE_TIME, STOP_WRITE_TIME, START_TRANS_TIME, STOP_TRANS_TIME, START_REPACK_TIME, STOP_REPACK_TIME, HLTKEY, LAST_UPDATE_TIME, DENSE_RANK() OVER (ORDER BY RUNNUMBER DESC NULLS LAST) runRank 
FROM SM_SUMMARY)
WHERE runRank <= 120
GROUP BY RUNNUMBER
ORDER BY RUNNUMBER DESC);


grant select on V_SM_SUMMARY to public;
