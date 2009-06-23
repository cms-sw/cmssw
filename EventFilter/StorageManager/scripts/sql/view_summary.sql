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

CREATE OR REPLACE FUNCTION TRANS_RATE_CHECK ( lastWrite in DATE, writeRate in Number, transRate in Number)
   RETURN NUMBER AS 
   status NUMBER;
BEGIN
        status := 0;
	IF (ABS(time_diff(sysdate, lastWrite)) < 180) THEN --Run is on
		IF (ABS(writeRate - transRate) / writeRate) > .15 AND writeRate > 10 THEN 
			status := 1;
		END IF;
		IF (ABS(writeRate - transRate) / writeRate) > .30 AND writeRate > 50 THEN
			status := 2;
		END IF;
	ELSE --Run is off
		IF (ABS(writeRate - transRate) / writeRate) > .15 AND writeRate > 10 THEN 
			status := 1;
		END IF;
		IF (ABS(writeRate - transRate) / writeRate) > .30 AND writeRate > 50 THEN
			status := 2;
		END IF;
	END IF;		
	
	RETURN status;			
END TRANS_RATE_CHECK;
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
	   "N_OPEN_STATUS",
	   "SAFE99_STATUS",
	   "SAFE0_STATUS" 
FROM ( SELECT TO_CHAR ( RUNNUMBER ) AS RUN_NUMBER,
	      TO_CHAR ( START_WRITE_TIME, 'dd.mm.yyyy hh24:mi:ss' ) AS START_TIME,
	      TO_CHAR ( LAST_UPDATE_TIME, 'dd.mm.yyyy hh24:mi:ss' ) AS UPDATE_TIME,
	      TO_CHAR ( setupLabel ) AS SETUPLABEL,
	      TO_CHAR ( APP_VERSION ) AS APP_VERSION,
	      TO_CHAR ( M_INSTANCE + 1 ) AS M_INSTANCE,
	      TO_CHAR ( ROUND (s_filesize/1073741824, 2) ) AS TOTAL_SIZE,
	      TO_CHAR ( NVL(s_Created, 0) ) AS NFILES,
	     (CASE s_injected
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((s_filesize2D / 1048576) / (GREATEST(time_diff( STOP_WRITE_TIME, START_WRITE_TIME),1)), 2))
	      END) AS RATE2D_AVG,  
	     (CASE s_copied
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((s_filesize2T0 / 1048576) / (GREATEST(time_diff( STOP_TRANS_TIME, START_TRANS_TIME),1)), 2))
	      END) AS RATE2T_AVG,	      
	      TO_CHAR ( s_NEvents ) AS NEVTS, 
	      TO_CHAR ( NVL(( s_Created ) - ( s_Injected ), 0) ) AS N_OPEN, 
	      TO_CHAR ( NVL(s_Injected, 0) ) AS N_CLOSED, 
	      TO_CHAR ( NVL(s_New, 0) ) AS N_SAFE0, 
	      TO_CHAR ( NVL(s_Checked, 0) ) AS N_SAFE99, 
	      TO_CHAR ( NVL(s_Deleted, 0) ) AS N_DELETED, 
	      TO_CHAR ( NVL(s_Repacked, 0) ) AS N_REPACKED,
	      TO_CHAR ( HLTKEY ) AS HLTKEY,
	      TO_CHAR ( OPEN_STATUS(STOP_WRITE_TIME, S_CREATED, S_INJECTED) ) AS N_OPEN_STATUS,
	      TO_CHAR ( SAFE0_CHECK(s_NEW, s_injected) ) AS SAFE0_STATUS,
	      TO_CHAR ( SAFE99_CHECK(STOP_WRITE_TIME, STOP_TRANS_TIME) ) AS SAFE99_STATUS
FROM ( SELECT *
        FROM SM_SUMMARY 
	ORDER BY RUNNUMBER DESC NULLS LAST)
	WHERE ROWNUM <= 120 );


grant select on V_SM_SUMMARY to public;
