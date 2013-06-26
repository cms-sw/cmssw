--
--provides per stream information (one row per stream each run)
--utilizes many of the functions declared in view_summary.sql
create or replace view view_sm_streams_full
AS SELECT  "RUN_NUMBER",
	   "STREAM", 
	   "START_TIME",
	   "UPDATE_TIME",
           "SETUPLABEL",
	   "APP_VERSION",
	   "M_INSTANCE", 
	   "TOTAL_SIZE",
	   "NFILES", 
	   "NEVTS",
           "EVTS_PER_FILE",
	   "EVT_RATE",
	   "EVT_AVG",
	   "RATE2D_AVG",
	   "RATE2T_AVG", 
	   "N_OPEN", 
	   "N_CLOSED",
	   "N_INJECTED",
           "N_TRANSFERRED", 
	   "N_CHECKED", 
	   "N_REPACKED",
	   "N_DELETED",
	   "HLTKEY",
	   "SETUP_STATUS",
	   "STREAM_STATUS",
	   "NEVT_STATUS",
	   "N_OPEN_STATUS",
	   "INJECTED_STATUS",
	   "WRITE_STATUS",
	   "TRANS_STATUS",
	   "TRANSFERRED_STATUS",
	   "CHECKED_STATUS",
           "REPACKED_STATUS",
           "DELETED_STATUS",
           "RANK"
FROM ( SELECT TO_CHAR ( RUNNUMBER ) AS RUN_NUMBER,
	      TO_CHAR ( STREAM ) AS STREAM,
	      TO_CHAR ( START_WRITE_TIME, 'dd.mm.yyyy hh24:mi:ss' ) AS START_TIME,
	      TO_CHAR ( LAST_UPDATE_TIME, 'dd.mm hh24:mi' ) AS UPDATE_TIME,
	      TO_CHAR ( setupLabel ) AS SETUPLABEL,
	      TO_CHAR ( APP_VERSION ) AS APP_VERSION,
	      TO_CHAR ( NVL(N_INSTANCE, M_INSTANCE + 1) ) AS M_INSTANCE,
	     (CASE NVL(s_filesize, 0)
                WHEN 0 THEN TO_CHAR('NA')
                ELSE TO_CHAR ( ROUND (s_filesize/1073741824, 2) )
              END) AS TOTAL_SIZE,
	      TO_CHAR ( NVL(s_Created, 0) ) AS NFILES,
	     (CASE NVL(s_injected, 0)
		WHEN 0 THEN TO_CHAR('NA')
		ELSE TO_CHAR ( ROUND ((s_filesize2D / 1048576) / (GREATEST(time_diff( STOP_WRITE_TIME, START_WRITE_TIME),1)), 1))
	      END) AS RATE2D_AVG,  
	     (CASE NVL(s_copied, 0)
		WHEN 0 THEN TO_CHAR('NA')
		ELSE TO_CHAR ( ROUND ((s_filesize2T0 / 1048576) / (GREATEST(time_diff( STOP_TRANS_TIME, START_TRANS_TIME),1)), 1))
	      END) AS RATE2T_AVG,	      
	     (CASE NVL(s_NEvents, 0)
                WHEN 0 THEN TO_CHAR('NA')
                ELSE  TO_CHAR ( s_NEvents )
              END) AS NEVTS, 
	     (CASE NVL(s_NEVENTS,0)
                WHEN 0 THEN TO_CHAR('NA')
                ELSE TO_CHAR( ROUND(s_NEVENTS / s_Created, 2) )
              END) AS EVTS_PER_FILE,
             (CASE NVL(s_NEVENTS, 0)
		WHEN 0 then TO_CHAR('NA')
		ELSE TO_CHAR ( ROUND (s_NEVENTS / (GREATEST(time_diff(STOP_WRITE_TIME, START_WRITE_TIME), 1)), 1))
	       END) AS EVT_RATE,
	     (CASE NVL(s_filesize, 0)
                WHEN 0 THEN TO_CHAR('NA')
                ELSE TO_CHAR ( ROUND ( (s_filesize / s_NEVENTS)/ 1024 , 1 ) )
              END) AS EVT_AVG,
	      TO_CHAR ( NVL( s_Created, 0 ) - NVL( s_Injected, 0) ) AS N_OPEN, 
	      TO_CHAR ( NVL(s_Injected, 0) ) AS N_CLOSED, 
	      TO_CHAR ( NVL(s_New,      0) ) AS N_INJECTED,
              TO_CHAR ( NVL(s_Copied,   0) ) AS N_TRANSFERRED, 
	      TO_CHAR ( NVL(s_Checked,  0) ) AS N_CHECKED, 
	      TO_CHAR ( NVL(s_Repacked, 0) ) AS N_REPACKED,
	      TO_CHAR ( NVL(s_Deleted,  0) ) AS N_DELETED, 
	      TO_CHAR ( HLTKEY ) AS HLTKEY,
	      (CASE
                WHEN setupLabel LIKE '%TransferTest%' THEN TO_CHAR(3)
                ELSE TO_CHAR(0)
              END) AS SETUP_STATUS,
              (CASE  
                WHEN  stream LIKE '%Error%'          THEN TO_CHAR(1)
                WHEN  stream LIKE '%_DontNotifyT0%'  THEN TO_CHAR(2)
                WHEN  stream LIKE '%_NoTransfer%'    THEN TO_CHAR(2)
                WHEN  stream LIKE '%_TransferTest%'  THEN TO_CHAR(3)
                WHEN  stream LIKE '%_NoRepack%'      THEN TO_CHAR(2)
                ELSE                                      TO_CHAR(0)
                END ) AS   STREAM_STATUS,           
              TO_CHAR ( NEVNTS_CHECK( RUNNUMBER, STREAM, s_NEvents ) ) AS NEVT_STATUS,
	      TO_CHAR ( OPEN_STATUS(NVL(STOP_WRITE_TIME, LAST_UPDATE_TIME), S_CREATED, S_INJECTED) ) AS N_OPEN_STATUS,
	     (CASE 
		WHEN (CASE NVL(s_injected,0)
		        WHEN 0 THEN 0
		        ELSE ROUND ((s_filesize2D / 1048576) / (GREATEST(time_diff( STOP_WRITE_TIME, START_WRITE_TIME),1)), 2)
	              END) < 2000 THEN TO_CHAR(0)
	        ELSE TO_CHAR(1)
	      END) AS WRITE_STATUS,
	      TO_CHAR ( TRANS_RATE_CHECK(STOP_WRITE_TIME,
		                        (CASE s_injected
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((s_filesize2D / 1048576) / (GREATEST(time_diff( STOP_WRITE_TIME, START_WRITE_TIME),1)), 2)
	                                 END),
		                        (CASE s_copied
		                         WHEN 0 THEN 0
		                         ELSE ROUND ((s_filesize2T0 / 1048576) / (GREATEST(time_diff( STOP_TRANS_TIME, START_TRANS_TIME),1)), 2)
	                                 END),
					 START_WRITE_TIME,
					 ROUND (s_filesize/1073741824, 2), 2) ) AS TRANS_STATUS,
	      TO_CHAR ( INJECTED_CHECK(   s_NEW,           s_injected,       s_Deleted,         STOP_WRITE_TIME,   NVL(N_INSTANCE, M_INSTANCE+1) ) ) AS INJECTED_STATUS,
	      TO_CHAR ( TRANSFERRED_CHECK(STOP_WRITE_TIME, STOP_TRANS_TIME,  NVL(s_NEW,0),      NVL(s_copied, 0),  NVL(N_INSTANCE, M_INSTANCE+1) ) ) AS TRANSFERRED_STATUS,
              TO_CHAR ( CHECKED_CHECK(    STOP_WRITE_TIME, STOP_TRANS_TIME,  NVL(s_checked,0),  NVL(s_copied, 0),  NVL(N_INSTANCE, M_INSTANCE+1) ) ) AS CHECKED_STATUS, 
	      TO_CHAR ( REPACKED_CHECK(   STOP_WRITE_TIME, STOP_TRANS_TIME,  NVL(s_REPACKED,0), NVL(s_NOTREPACKED,0), NVL(s_CHECKED,0),  NVL(s_Deleted,0)                        ) ) AS REPACKED_STATUS,
              TO_CHAR ( DELETED_CHECK(    N_INSTANCE,      START_WRITE_TIME,  NVL(s_created,0),  NVL(s_injected,0), NVL(s_Checked,0),  NVL(s_Repacked,0), NVL(s_NotRepacked,0),  NVL(s_Deleted, 0), STOP_TRANS_TIME ) ) AS DELETED_STATUS, 
	      TO_CHAR ( dr ) AS RANK
FROM ( SELECT runnumber, stream, start_write_time, last_update_time, setuplabel, app_version, n_instance, m_instance, s_filesize, s_created, 
              s_filesize2d, s_filesize2T0, s_NEvents, s_injected, s_new, s_copied, s_checked, s_deleted, s_repacked, s_notrepacked, HLTKEY, 
              STOP_WRITE_TIME, start_trans_time, STOP_TRANS_TIME, DENSE_RANK() OVER ( ORDER BY RUNNUMBER DESC NULLS LAST) dr
        FROM SM_SUMMARY)
	where dr <= 30)
ORDER BY 1 DESC , 2 ASC;

grant select on view_sm_streams_full to public;
