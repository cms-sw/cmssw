-- First the Inject Worker ones
CREATE OR REPLACE PROCEDURE FILES_CREATED_PROC_SUMMARY (
    v_filename IN Varchar
)
IS
v_producer    VARCHAR2(100);
v_stream      VARCHAR2(100);
v_instance    NUMBER(5);
v_runnumber   NUMBER(10);
v_lumisection NUMBER(10);
v_setuplabel  VARCHAR2(100);
v_app_version VARCHAR2(100);
v_timestamp   TIMESTAMP(6);
v_etime       VARCHAR2(64);
v_nrows       NUMBER(1);
BEGIN
     SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER, LUMISECTION, SETUPLABEL, APP_VERSION, CTIME
     into v_producer, v_stream, v_instance, v_runnumber, v_lumisection, v_setuplabel, v_app_version, v_timestamp
     FROM FILES_CREATED
     WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        v_nrows := 0;
        SELECT COUNT(RUNNUMBER) into v_nrows  from SM_SUMMARY  WHERE RUNNUMBER = v_runnumber AND STREAM= v_stream;

        IF  v_nrows = 0 THEN
             LOCK TABLE SM_SUMMARY  IN EXCLUSIVE MODE;  
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || '  14-FILES_CREATED_AI: Established LOCK  SM_SUMMARY, do MERGE  <<');
            --try again with lock now in place:
 	     MERGE INTO SM_SUMMARY
             using dual on (RUNNUMBER = v_runnumber   AND STREAM= v_stream)
             when matched then update 
 	       SET S_LUMISECTION = NVL(S_LUMISECTION,0) + NVL(v_lumisection,0),
                    S_CREATED = NVL(S_CREATED,0) + 1,
 	            M_INSTANCE = GREATEST(v_instance, NVL(M_INSTANCE, 0)),
 		    START_WRITE_TIME =  LEAST(v_timestamp, NVL(START_WRITE_TIME,v_timestamp)),
 		    LAST_UPDATE_TIME = sysdate
               when not matched then 
	   	INSERT ( RUNNUMBER, STREAM, SETUPLABEL, APP_VERSION, S_LUMISECTION,
	            S_CREATED, N_INSTANCE, M_INSTANCE, START_WRITE_TIME, LAST_UPDATE_TIME)
                  VALUES ( v_runnumber, v_stream, v_setuplabel, v_app_version, v_lumisection,
         	    1, 1, v_instance, v_timestamp, sysdate);
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || '  15-FILES_CREATED_AI: done LOCK/INSERT SM_SUMMARY   SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' SM_SUMMARY  FILE: '|| v_filename || '   <<');
 
        ELSE
             UPDATE SM_SUMMARY
                   SET S_LUMISECTION = NVL(S_LUMISECTION,0) + NVL(v_lumisection,0),
                       S_CREATED = NVL(S_CREATED,0) + 1,
                       M_INSTANCE = GREATEST(v_instance, NVL(M_INSTANCE, 0)),
                       START_WRITE_TIME =  LEAST(v_timestamp, NVL(START_WRITE_TIME,v_timestamp)),
                       LAST_UPDATE_TIME = sysdate
                   WHERE RUNNUMBER = v_runnumber AND STREAM= v_stream;
        END IF;
        
        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || ' -FILES_CREATED: done  SM_SUMMARY   FILE: '|| v_filename || '   <<');
   END IF;
END;
/
GRANT execute on FILES_CREATED_PROC_SUMMARY to CMS_STOMGR_W;

-- Now do SM_INSTANCES
CREATE OR REPLACE PROCEDURE FILES_CREATED_PROC_INSTANCES (
    v_filename IN Varchar
)
IS
v_producer    VARCHAR2(100);
v_stream      VARCHAR2(100);
v_instance    NUMBER(5);
v_runnumber   NUMBER(10);
v_lumisection NUMBER(10);
v_setuplabel  VARCHAR2(100);
v_hostname    VARCHAR2(100);
v_timestamp   TIMESTAMP(6);
v_etime       VARCHAR2(64);
v_nrows       NUMBER(1);
BEGIN
     SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER, LUMISECTION, SETUPLABEL, HOSTNAME, CTIME
     into v_producer, v_stream, v_instance, v_runnumber, v_lumisection, v_setuplabel, v_hostname, v_timestamp
     FROM FILES_CREATED
     WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        v_nrows := 0;
        SELECT COUNT(RUNNUMBER) into v_nrows  from SM_INSTANCES WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance ;

        IF v_nrows = 0 THEN
             LOCK TABLE SM_INSTANCES  IN EXCLUSIVE MODE;  
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || ' 114-FILES_CREATED_AI:  Established LOCK  SM_INSTANCES, do MERGE  <<');
  	     MERGE INTO  SM_INSTANCES
                using dual on (RUNNUMBER = v_runnumber  AND INSTANCE = v_instance )
                when matched then update 
                   SET N_CREATED = NVL(N_CREATED,0) + 1
                when not matched then 
	   	   INSERT ( RUNNUMBER, INSTANCE, HOSTNAME, N_CREATED, START_WRITE_TIME, SETUPLABEL)
                    VALUES ( v_runnumber, v_instance, v_hostname, 1, v_timestamp, v_setuplabel);
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || ' 115-FILES_CREATED_AI: Done LOCK/INSERT SM_INSTANCES SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' SM_SUMMARY  FILE: '|| v_filename || '   <<');

        ELSE
             UPDATE SM_INSTANCES
               SET N_CREATED = NVL(N_CREATED,0) + 1
               WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
        END IF;

        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || ' -FILES_CREATED: done  SM_INSTANCES FILE: '|| v_filename || ' -----------  <<');
   END IF;
END;
/

GRANT execute on FILES_CREATED_PROC_INSTANCES to CMS_STOMGR_W;

-- Files injected, when they're closed
CREATE OR REPLACE PROCEDURE FILES_INJECTED_PROC_SUMMARY (
    v_filename IN Varchar
)
IS
v_producer    VARCHAR2(100);
v_stream      VARCHAR2(100);
v_instance    NUMBER(5);
v_runnumber   NUMBER(10);
v_nevents     NUMBER(20);
v_filesize    NUMBER(20);
v_comment_str VARCHAR2(1000);
v_timestamp   TIMESTAMP(6);
v_etime       VARCHAR2(64);
BEGIN
     SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
     into v_producer, v_stream, v_instance, v_runnumber
     FROM FILES_CREATED WHERE FILENAME = v_filename;

     IF v_producer = 'StorageManager' THEN
         SELECT NEVENTS, FILESIZE, COMMENT_STR, ITIME
         into v_nevents, v_filesize, v_comment_str, v_timestamp
         FROM FILES_INJECTED
         WHERE FILENAME = v_filename;

       	 UPDATE SM_SUMMARY
                 SET S_NEVENTS = NVL(S_NEVENTS,0) + NVL(v_nevents,0),
            	 S_FILESIZE = NVL(S_FILESIZE,0) + NVL(v_filesize,0),
            	 S_FILESIZE2D = NVL(S_FILESIZE2D,0) + NVL(v_filesize,0),
            	 S_INJECTED = NVL(S_INJECTED,0) + 1,
                 N_INSTANCE = (SELECT COUNT(DISTINCT INSTANCE) FROM FILES_CREATED WHERE RUNNUMBER = v_runnumber AND STREAM = v_stream),
	    	 STOP_WRITE_TIME = GREATEST(v_timestamp, NVL(STOP_WRITE_TIME, v_timestamp)),
	    	 HLTKEY = NVL(HLTKEY, v_comment_str),
            	 LAST_UPDATE_TIME = sysdate
      	 WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
         DBMS_OUTPUT.PUT_LINE ( v_etime || ' -FILES_INJECTED: done  SM_SUMMARY   FILE: '|| v_filename || '   <<');
     END IF;
END;
/

GRANT execute on FILES_INJECTED_PROC_SUMMARY to CMS_STOMGR_W;

CREATE OR REPLACE PROCEDURE FILES_INJECTED_PROC_INSTANCES (
    v_filename IN Varchar
)
IS
v_producer    VARCHAR2(100);
v_instance    NUMBER(5);
v_runnumber   NUMBER(10);
v_timestamp   TIMESTAMP(6);
v_etime       VARCHAR2(64);
BEGIN
     SELECT PRODUCER, INSTANCE, RUNNUMBER
     into v_producer, v_instance, v_runnumber
     FROM FILES_CREATED WHERE FILENAME = v_filename;

     IF v_producer = 'StorageManager' THEN
         SELECT ITIME into v_timestamp FROM FILES_INJECTED WHERE FILENAME = v_filename;

          v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
          DBMS_OUTPUT.PUT_LINE (v_etime || '  3-FILES_INJECTED:  pre-UPDATE INSTANCES   <<');
          UPDATE SM_INSTANCES
                  SET N_INJECTED = NVL(N_INJECTED,0) + 1,
                  LAST_WRITE_TIME = GREATEST(v_timestamp, NVL(LAST_WRITE_TIME, v_timestamp))
          WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
          v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
          DBMS_OUTPUT.PUT_LINE ( v_etime || ' -FILES_INJECTED: done  SM_INSTANCES FILE: '|| v_filename || ' -----------  <<');
     END IF;
END;
/

GRANT execute on FILES_INJECTED_PROC_INSTANCES to CMS_STOMGR_W;

-- ###
-- Now the Tier0 / Copy Manager - Transfer Status Worker one
-- ###

-- File has been sent to the Copy Manager


CREATE OR REPLACE PROCEDURE TRANS_NEW_PROC_SUMMARY (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
v_timestamp TIMESTAMP(6);
BEGIN
    SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
    into v_producer, v_stream, v_instance, v_runnumber
    FROM FILES_CREATED
    WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        SELECT ITIME into v_timestamp from FILES_TRANS_NEW where FILENAME = v_filename;
   	UPDATE SM_SUMMARY
        SET S_NEW = NVL(S_NEW,0) + 1,
	    START_TRANS_TIME =  LEAST(v_timestamp, NVL(START_TRANS_TIME,v_timestamp)),
            LAST_UPDATE_TIME = sysdate
      	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
    END IF;
END;
/
GRANT execute on TRANS_NEW_PROC_SUMMARY to CMS_STOMGR_TIER0_W;

CREATE OR REPLACE PROCEDURE TRANS_NEW_PROC_INSTANCES (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
BEGIN
    SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
    into v_producer, v_stream, v_instance, v_runnumber
    FROM FILES_CREATED
    WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        UPDATE SM_INSTANCES
        SET N_NEW = NVL(N_NEW,0) + 1
        WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
    END IF;
END;
/
GRANT execute on TRANS_NEW_PROC_INSTANCES to CMS_STOMGR_TIER0_W;

-- File has been copied by the Copy Worker

CREATE OR REPLACE PROCEDURE TRANS_COPIED_PROC_SUMMARY (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
v_timestamp TIMESTAMP(6);
BEGIN
    SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
    into v_producer, v_stream, v_instance, v_runnumber
    FROM FILES_CREATED
    WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        SELECT ITIME into v_timestamp from FILES_TRANS_COPIED where FILENAME = v_filename;
     	UPDATE SM_SUMMARY
        SET S_COPIED = NVL(S_COPIED,0) + 1,
	    STOP_TRANS_TIME = GREATEST(v_timestamp, NVL(STOP_TRANS_TIME, v_timestamp)),
            S_FILESIZE2T0 = NVL(S_FILESIZE2T0,0) + 
		NVL((SELECT FILESIZE from FILES_INJECTED where FILENAME = v_filename),0),
            LAST_UPDATE_TIME = sysdate
      	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
    END IF;
END;
/
GRANT execute on TRANS_COPIED_PROC_SUMMARY to CMS_STOMGR_TIER0_W;

CREATE OR REPLACE PROCEDURE TRANS_COPIED_PROC_INSTANCES (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
BEGIN
    SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
    into v_producer, v_stream, v_instance, v_runnumber
    FROM FILES_CREATED
    WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        UPDATE SM_INSTANCES
        SET N_COPIED = NVL(N_COPIED,0) + 1
        WHERE RUNNUMBER = v_runnumber AND INSTANCE=v_instance;
    END IF;
END;
/
GRANT execute on TRANS_COPIED_PROC_INSTANCES to CMS_STOMGR_TIER0_W;

-- File has been checked by Tier0

CREATE OR REPLACE PROCEDURE TRANS_CHECKED_PROC_SUMMARY (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
v_timestamp TIMESTAMP(6);
BEGIN
     SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
     into v_producer, v_stream, v_instance, v_runnumber
     FROM FILES_CREATED
     WHERE FILENAME = v_filename;
     IF v_producer = 'StorageManager' THEN
        SELECT ITIME into v_timestamp from FILES_TRANS_CHECKED where FILENAME = v_filename;
        UPDATE SM_SUMMARY
            SET S_CHECKED     = NVL(S_CHECKED,0)     + 1,
                S_NOTREPACKED = NVL(S_NOTREPACKED,0) + DECODE(v_stream, 'Error', 1, '%_NoRepack', 1, 0),
                START_REPACK_TIME = LEAST(v_timestamp, NVL(START_REPACK_TIME, v_timestamp)),
                LAST_UPDATE_TIME  = sysdate
        WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
    END IF;
END;
/
GRANT execute on TRANS_CHECKED_PROC_SUMMARY to CMS_STOMGR_TIER0_W;

CREATE OR REPLACE PROCEDURE TRANS_CHECKED_PROC_INSTANCES (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
v_timestamp TIMESTAMP(6);
BEGIN
     SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
     into v_producer, v_stream, v_instance, v_runnumber
     FROM FILES_CREATED
     WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        SELECT ITIME into v_timestamp from FILES_TRANS_CHECKED where FILENAME = v_filename;
        UPDATE SM_INSTANCES
        SET N_CHECKED     = NVL(N_CHECKED,0)     + 1,
            START_REPACK_TIME = LEAST(v_timestamp, NVL(START_REPACK_TIME, v_timestamp)),
            N_NOTREPACKED = NVL(N_NOTREPACKED,0) + DECODE(v_stream, 'Error', 1, '%_NoRepack', 1, 0)
        WHERE RUNNUMBER   = v_runnumber AND INSTANCE = v_instance;
    END IF;
END;
/
GRANT execute on TRANS_CHECKED_PROC_INSTANCES to CMS_STOMGR_TIER0_W;


-- File has been repacked by Tier0

CREATE OR REPLACE PROCEDURE TRANS_REPACKED_PROC_SUMMARY (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
v_timestamp TIMESTAMP(6);
BEGIN
    SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
    into v_producer, v_stream, v_instance, v_runnumber
    FROM FILES_CREATED
    WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        SELECT ITIME into v_timestamp from FILES_TRANS_COPIED where FILENAME = v_filename;
     	UPDATE SM_SUMMARY
        SET S_REPACKED = NVL(S_REPACKED,0) + 1,
	    STOP_REPACK_TIME = GREATEST(v_timestamp, NVL(STOP_REPACK_TIME, v_timestamp)),
	    LAST_UPDATE_TIME = sysdate
      	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
    END IF;
END;
/
GRANT execute on TRANS_REPACKED_PROC_SUMMARY to CMS_STOMGR_TIER0_W;


CREATE OR REPLACE PROCEDURE TRANS_REPACKED_PROC_INSTANCES (
    v_filename IN Varchar
)
IS
v_producer  VARCHAR2(100);
v_stream    VARCHAR2(100);
v_instance  NUMBER(5);
v_runnumber NUMBER(10);
v_timestamp TIMESTAMP(6);
BEGIN
    SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER
    into v_producer, v_stream, v_instance, v_runnumber
    FROM FILES_CREATED
    WHERE FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        SELECT ITIME into v_timestamp from FILES_TRANS_COPIED where FILENAME = v_filename;
        UPDATE SM_INSTANCES
        SET N_REPACKED = NVL(N_REPACKED,0) + 1,
	STOP_REPACK_TIME = GREATEST(v_timestamp, NVL(STOP_REPACK_TIME, v_timestamp))
        WHERE RUNNUMBER = v_runnumber AND INSTANCE=v_instance;
    END IF;
END;
/
GRANT execute on TRANS_REPACKED_PROC_INSTANCES to CMS_STOMGR_TIER0_W;

