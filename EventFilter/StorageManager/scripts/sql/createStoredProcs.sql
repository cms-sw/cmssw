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
        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || '   1-FILES_CREATED_AI: preQuery  SM_SUMMARY   FILE: '|| v_filename || '   <<');
        SELECT COUNT(RUNNUMBER) into v_nrows  from SM_SUMMARY  WHERE RUNNUMBER = v_runnumber AND STREAM= v_stream;
        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || '   2-FILES_CREATED_AI: postQuery  SM_SUMMARY  rows found: ' ||  v_nrows || '  <<');

        IF  v_nrows = 0 THEN
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || '  13-FILES_CREATED_AI: Query=0, try merge, initiate LOCK on  SM_SUMMARY  <<');

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
	   	INSERT (
		    RUNNUMBER,
	            STREAM,
	            SETUPLABEL,
	            APP_VERSION,
	            S_LUMISECTION,
	            S_CREATED,
	            N_INSTANCE,
	            M_INSTANCE,
	            START_WRITE_TIME,
        	    LAST_UPDATE_TIME)
                  VALUES (
       		    v_runnumber,
       		    v_stream,
  	            v_setuplabel,
        	    v_app_version,
            	    v_lumisection,
         	    1,
                    1,
          	    v_instance,
                    v_timestamp,
                    sysdate);
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || '  15-FILES_CREATED_AI: done LOCK/INSERT SM_SUMMARY   SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' SM_SUMMARY  FILE: '|| v_filename || '   <<');
 
        ELSE
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || '  23-FILES_CREATED_AI: PreUpdate,  on  SM_SUMMARY  <<');
             UPDATE SM_SUMMARY
                   SET S_LUMISECTION = NVL(S_LUMISECTION,0) + NVL(v_lumisection,0),
                       S_CREATED = NVL(S_CREATED,0) + 1,
                       M_INSTANCE = GREATEST(v_instance, NVL(M_INSTANCE, 0)),
                       START_WRITE_TIME =  LEAST(v_timestamp, NVL(START_WRITE_TIME,v_timestamp)),
                       LAST_UPDATE_TIME = sysdate
                   WHERE RUNNUMBER = v_runnumber AND STREAM= v_stream;
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || '  24-FILES_CREATED_AI: PostUpdate on  SM_SUMMARY   SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' <<');
        END IF;
        
        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || '  55-FILES_CREATED_AI: done  SM_SUMMARY   FILE: '|| v_filename || '   <<');
     DBMS_OUTPUT.PUT_LINE ('-------FILES_CREATED Done SM_SUMMARY:' ||v_filename || '-------------------' );
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
        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || '101-FILES_CREATED_AI: preQuery  SM_INSTANCES    FILE: '|| v_filename || '   <<');
        SELECT COUNT(RUNNUMBER) into v_nrows  from SM_INSTANCES WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance ;
        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || '102-FILES_CREATED_AI: postQuery  SM_INSTANCES  rows found: '||  v_nrows || '   <<');

        IF v_nrows = 0 THEN
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || ' 113-FILES_CREATED_AI: Q-Failed, try merge, initiate LOCK on  SM_INSTANCES  <<');

             LOCK TABLE SM_INSTANCES  IN EXCLUSIVE MODE;  
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || ' 114-FILES_CREATED_AI:  Established LOCK  SM_INSTANCES, do MERGE  <<');
  	     MERGE INTO  SM_INSTANCES
                using dual on (RUNNUMBER = v_runnumber  AND INSTANCE = v_instance )
                when matched then update 
                   SET N_CREATED = NVL(N_CREATED,0) + 1
                when not matched then 
	   	   INSERT (                
                      RUNNUMBER,
                      INSTANCE,
                      HOSTNAME, 
                      N_CREATED,
                      START_WRITE_TIME,
                      SETUPLABEL)
                    VALUES (
                      v_runnumber,
                      v_instance,
                      v_hostname,
                      1,
                      v_timestamp,
                      v_setuplabel);
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || ' 115-FILES_CREATED_AI: Done LOCK/INSERT SM_INSTANCES SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' SM_SUMMARY  FILE: '|| v_filename || '   <<');

        ELSE
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || ' 123-FILES_CREATED_AI: preUpdate  SM_INSTANCES  <<');
             UPDATE SM_INSTANCES
               SET N_CREATED = NVL(N_CREATED,0) + 1
               WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
             v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
             DBMS_OUTPUT.PUT_LINE ( v_etime || ' 124-FILES_CREATED_AI: postUpdate on  SM_INSTANCES  SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' <<');
        END IF;

        v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
        DBMS_OUTPUT.PUT_LINE ( v_etime || ' 155-FILES_CREATED_AI: done  SM_INSTANCES FILE: '|| v_filename || ' -----------  <<');
     DBMS_OUTPUT.PUT_LINE ('-------FILES_CREATED Done SM_INSTANCES:' ||v_filename || '-------------------' );
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

         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
         DBMS_OUTPUT.PUT_LINE ( v_etime || '  1-FILES_INJECTED: pre UPDATE  SM_SUMMARY    <<'); 
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
         SELECT ITIME
         into v_timestamp
         FROM FILES_INJECTED
         WHERE FILENAME = v_filename;

          v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
          DBMS_OUTPUT.PUT_LINE (v_etime || '  3-FILES_INJECTED:  pre-UPDATE INSTANCES   <<');
          UPDATE SM_INSTANCES
                  SET N_INJECTED = NVL(N_INJECTED,0) + 1,
                  LAST_WRITE_TIME = GREATEST(v_timestamp, NVL(LAST_WRITE_TIME, v_timestamp))
          WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
          v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
          DBMS_OUTPUT.PUT_LINE (v_etime || '  5-FILES_INJECTED: ALL done!  FILE: '|| v_filename || ' -------   <<');
     END IF;
END;
/

GRANT execute on FILES_INJECTED_PROC_INSTANCES to CMS_STOMGR_W;


-- Now the Tier0 / Copy Manager - Transfer Status Worker one
CREATE OR REPLACE PROCEDURE FILES_TRANS_CHECKED_PROC (
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

    SELECT ITIME
    into v_timestamp
    from FILES_TRANS_CHECKED
    where FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
        UPDATE SM_SUMMARY
                SET S_CHECKED = NVL(S_CHECKED,0) + 1,
                START_REPACK_TIME = LEAST(v_timestamp, NVL(START_REPACK_TIME, v_timestamp)),
                LAST_UPDATE_TIME = sysdate
        WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
        IF SQL%ROWCOUNT = 0 THEN
                NULL;
        END IF;
        UPDATE SM_INSTANCES
                SET N_CHECKED = NVL(N_CHECKED,0) + 1
        WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
        IF SQL%ROWCOUNT = 0 THEN
                NULL;
        END IF;
    END IF;
END;
/
GRANT execute on FILES_TRANS_CHECKED_PROC to CMS_STOMGR_TIER0_W;

CREATE OR REPLACE PROCEDURE FILES_TRANS_NEW_PROC (
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

    SELECT ITIME
    into v_timestamp
    from FILES_TRANS_NEW
    where FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
   	UPDATE SM_SUMMARY
        	SET S_NEW = NVL(S_NEW,0) + 1,
	    	START_TRANS_TIME =  LEAST(v_timestamp, NVL(START_TRANS_TIME,v_timestamp)),
            	LAST_UPDATE_TIME = sysdate
      	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
     	IF SQL%ROWCOUNT = 0 THEN
         	NULL;
     	END IF;
        UPDATE SM_INSTANCES
                SET N_NEW = NVL(N_NEW,0) + 1
        WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
        IF SQL%ROWCOUNT = 0 THEN
                NULL;
        END IF;
     END IF;
 END;
/
GRANT execute on FILES_TRANS_NEW_PROC to CMS_STOMGR_TIER0_W;

CREATE OR REPLACE PROCEDURE FILES_TRANS_COPIED_PROC (
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

    SELECT ITIME
    into v_timestamp
    from FILES_TRANS_COPIED
    where FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
     	UPDATE SM_SUMMARY
        	SET S_COPIED = NVL(S_COPIED,0) + 1,
	    	STOP_TRANS_TIME = GREATEST(v_timestamp, NVL(STOP_TRANS_TIME, v_timestamp)),
            	S_FILESIZE2T0 = NVL(S_FILESIZE2T0,0) + 
			NVL((SELECT FILESIZE from FILES_INJECTED where FILENAME = v_filename),0),
            	LAST_UPDATE_TIME = sysdate
      	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
     	IF SQL%ROWCOUNT = 0 THEN
         	NULL;
     	END IF;
        UPDATE SM_INSTANCES
                SET N_COPIED = NVL(N_COPIED,0) + 1
        WHERE RUNNUMBER = v_runnumber AND INSTANCE=v_instance;
        IF SQL%ROWCOUNT = 0 THEN
                NULL;
        END IF;
     END IF;
END;
/
GRANT execute on FILES_TRANS_COPIED_PROC to CMS_STOMGR_TIER0_W;

CREATE OR REPLACE PROCEDURE FILES_TRANS_REPACKED_PROC (
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

    SELECT ITIME
    into v_timestamp
    from FILES_TRANS_COPIED
    where FILENAME = v_filename;

    IF v_producer = 'StorageManager' THEN
     	UPDATE SM_SUMMARY
        	SET S_REPACKED = NVL(S_REPACKED,0) + 1,
	    	STOP_REPACK_TIME = v_timestamp,
	    	LAST_UPDATE_TIME = sysdate
      	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
     	IF SQL%ROWCOUNT = 0 THEN
        	 NULL;
     	END IF;
        UPDATE SM_INSTANCES
                SET N_REPACKED = NVL(N_REPACKED,0) + 1
        WHERE RUNNUMBER = v_runnumber AND INSTANCE=v_instance;
        IF SQL%ROWCOUNT = 0 THEN
                NULL;
        END IF;
     END IF;
END;
/

GRANT execute on FILES_TRANS_REPACKED_PROC to CMS_STOMGR_TIER0_W;
