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
