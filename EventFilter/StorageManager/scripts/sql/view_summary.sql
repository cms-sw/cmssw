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
	   "HLTKEY" 
FROM ( SELECT TO_CHAR ( RUNNUMBER ) AS RUN_NUMBER,
	      TO_CHAR ( START_WRITE_TIME, 'dd.mm.yyyy hh24:mi:ss' ) AS START_TIME,
	      TO_CHAR ( LAST_UPDATE_TIME, 'dd.mm.yyyy hh24:mi:ss' ) AS UPDATE_TIME,
	      TO_CHAR ( setupLabel ) AS SETUPLABEL,
	      TO_CHAR ( APP_VERSION ) AS APP_VERSION,
	      TO_CHAR ( M_INSTANCE + 1 ) AS M_INSTANCE,
	      TO_CHAR ( ROUND (s_filesize/1073741824, 2) ) AS TOTAL_SIZE,
	      TO_CHAR ( s_Created ) AS NFILES,
	     (CASE s_injected
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((s_filesize2D / 1048576) / (GREATEST(time_diff( STOP_WRITE_TIME, START_WRITE_TIME),1)), 2))
	      END) AS RATE2D_AVG,  
	     (CASE s_copied
		WHEN 0 THEN TO_CHAR(0)
		ELSE TO_CHAR ( ROUND ((s_filesize2T0 / 1048576) / (GREATEST(time_diff( STOP_TRANS_TIME, START_TRANS_TIME),1)), 2))
	      END) AS RATE2T_AVG,	      
	      TO_CHAR ( s_NEvents ) AS NEVTS, 
	      TO_CHAR ( ( s_Created ) - ( s_Injected ) ) AS N_OPEN, 
	      TO_CHAR ( s_Injected ) AS N_CLOSED, 
	      TO_CHAR ( s_New ) AS N_SAFE0, 
	      TO_CHAR ( s_Checked ) AS N_SAFE99, 
	      TO_CHAR ( s_Deleted ) AS N_DELETED, 
	      TO_CHAR ( s_Repacked ) AS N_REPACKED,
	      TO_CHAR ( HLTKEY ) AS HLTKEY 
FROM ( SELECT *
        FROM SM_SUMMARY 
	ORDER BY RUNNUMBER DESC NULLS LAST)
	WHERE ROWNUM <= 120 );


grant select on V_SM_SUMMARY to public;
