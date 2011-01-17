CREATE OR REPLACE PROCEDURE "CMS_STOMGR"."FILES_TRANS_CHECKED_PROC" (
    v_filename IN Varchar
)
IS
v_producer VARCHAR(30);
v_stream VARCHAR(30);
v_instance NUMBER(5);
v_runnumber NUMBER(10);
BEGIN
     SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER into v_producer, v_stream, v_instance, v_runnumber FROM FILES_CREATED WHERE FILENAME = v_filename;
     IF v_producer = 'StorageManager' THEN
        UPDATE SM_SUMMARY
                SET S_CHECKED = NVL(S_CHECKED,0) + 1,
                START_REPACK_TIME = LEAST(CURRENT_TIMESTAMP, NVL(START_REPACK_TIME, CURRENT_TIMESTAMP)),
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

