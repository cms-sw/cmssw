-- Migrated to a stored procedure
-- CREATE OR REPLACE TRIGGER FILES_CREATED_AI
-- AFTER INSERT ON FILES_CREATED
-- FOR EACH ROW
-- DECLARE
--    v_code  NUMBER;
--    v_errm  VARCHAR2(64);
--    v_etime VARCHAR2(64);
--    v_nrows  NUMBER;
-- BEGIN
--    IF :NEW.PRODUCER = 'StorageManager' THEN
--       BEGIN
-- 
--         v_nrows := 0;
--         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--         DBMS_OUTPUT.PUT_LINE ( v_etime || '   1-FILES_CREATED_AI: preQuery  SM_SUMMARY   FILE: '|| :NEW.FILENAME || '   <<');
--       
--         SELECT COUNT(RUNNUMBER) into v_nrows  from SM_SUMMARY  WHERE RUNNUMBER = :NEW.RUNNUMBER AND STREAM= :NEW.STREAM;
--  
--         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--         DBMS_OUTPUT.PUT_LINE ( v_etime || '   2-FILES_CREATED_AI: postQuery  SM_SUMMARY  rows found: ' ||  v_nrows || '  <<');
-- 
-- 
--         IF  v_nrows = 0 THEN
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || '  13-FILES_CREATED_AI: Query=0, try merge, initiate LOCK on  SM_SUMMARY  <<');
-- 
--              LOCK TABLE SM_SUMMARY  IN EXCLUSIVE MODE;  
-- 
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || '  14-FILES_CREATED_AI: Established LOCK  SM_SUMMARY, do MERGE  <<');
--  
--             --try again with lock now in place:
--  	     MERGE INTO SM_SUMMARY
--              using dual on (RUNNUMBER = :NEW.RUNNUMBER   AND STREAM= :NEW.STREAM)
--              when matched then update 
--  	       SET S_LUMISECTION = NVL(S_LUMISECTION,0) + NVL(:NEW.LUMISECTION,0),
--                     S_CREATED = NVL(S_CREATED,0) + 1,
--  	            M_INSTANCE = GREATEST(:NEW.INSTANCE, NVL(M_INSTANCE, 0)),
--  		    START_WRITE_TIME =  LEAST(:NEW.CTIME, NVL(START_WRITE_TIME,:NEW.CTIME)),
--  		    LAST_UPDATE_TIME = sysdate
--                when not matched then 
-- 	   	INSERT (
-- 		    RUNNUMBER,
-- 	            STREAM,
-- 	            SETUPLABEL,
-- 	            APP_VERSION,
-- 	            S_LUMISECTION,
-- 	            S_CREATED,
-- 	            N_INSTANCE,
-- 	            M_INSTANCE,
-- 	            START_WRITE_TIME,
--         	    LAST_UPDATE_TIME)
--                   VALUES (
--        		    :NEW.RUNNUMBER,
--        		    :NEW.STREAM,
--   	            :NEW.SETUPLABEL,
--         	    :NEW.APP_VERSION,
--             	    :NEW.LUMISECTION,
--          	    1,
--                     1,
--           	    :NEW.INSTANCE,
--                     :NEW.CTIME,
--                     sysdate);
--    
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || '  15-FILES_CREATED_AI: done LOCK/INSERT SM_SUMMARY   SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' SM_SUMMARY  FILE: '|| :NEW.FILENAME || '   <<');
--  
-- 
--         ELSE
-- 
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || '  23-FILES_CREATED_AI: PreUpdate,  on  SM_SUMMARY  <<');
-- 
--              UPDATE SM_SUMMARY
--                    SET S_LUMISECTION = NVL(S_LUMISECTION,0) + NVL(:NEW.LUMISECTION,0),
--                        S_CREATED = NVL(S_CREATED,0) + 1,
--                        M_INSTANCE = GREATEST(:NEW.INSTANCE, NVL(M_INSTANCE, 0)),
--                        START_WRITE_TIME =  LEAST(:NEW.CTIME, NVL(START_WRITE_TIME,:NEW.CTIME)),
--                        LAST_UPDATE_TIME = sysdate
--                    WHERE RUNNUMBER = :NEW.RUNNUMBER AND STREAM= :NEW.STREAM;
-- 
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || '  24-FILES_CREATED_AI: PostUpdate on  SM_SUMMARY   SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' <<');
-- 
--         END IF;
--         
--         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--         DBMS_OUTPUT.PUT_LINE ( v_etime || '  55-FILES_CREATED_AI: done  SM_SUMMARY   FILE: '|| :NEW.FILENAME || '   <<');
--   
-- 
-- 
--         EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_INJECTED so handle it with message:
--             WHEN DUP_VAL_ON_INDEX THEN
--                DBMS_OUTPUT.PUT_LINE ('FILES_CREATED_AI  DUP_VAL_ON_INDEX for SM_SUMMARY; FILE: ' || :NEW.FILENAME || ' << ');
--                WHEN OTHERS THEN
--                    v_code := SQLCODE;
--                    v_errm := SUBSTR(SQLERRM, 1 , 64);
--                    v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--                    DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_CREATED_AI  for SM_SUMMARY: ' ||  v_errm || ' << ');
--                    DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_CREATED_AI  for SM_SUMMARY; FILE: ' || :NEW.FILENAME || '  <<');
--         
-- 
--       END;
-- 
-- 
-- -----DO SM_INSTANCES:
-- 
--       BEGIN
--         v_nrows := 0;
--         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--         DBMS_OUTPUT.PUT_LINE ( v_etime || '101-FILES_CREATED_AI: preQuery  SM_INSTANCES    FILE: '|| :NEW.FILENAME || '   <<');
--  
--         SELECT COUNT(RUNNUMBER) into v_nrows  from SM_INSTANCES WHERE RUNNUMBER = :NEW.RUNNUMBER AND INSTANCE = :NEW.INSTANCE ;
--  
--         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--         DBMS_OUTPUT.PUT_LINE ( v_etime || '102-FILES_CREATED_AI: postQuery  SM_INSTANCES  rows found: '||  v_nrows || '   <<');
-- 
-- 
--         IF v_nrows = 0 THEN
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || ' 113-FILES_CREATED_AI: Q-Failed, try merge, initiate LOCK on  SM_INSTANCES  <<');
-- 
--              LOCK TABLE SM_INSTANCES  IN EXCLUSIVE MODE;  
-- 
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || ' 114-FILES_CREATED_AI:  Established LOCK  SM_INSTANCES, do MERGE  <<');
-- 
--   	     MERGE INTO  SM_INSTANCES
--                 using dual on (RUNNUMBER = :NEW.RUNNUMBER  AND INSTANCE = :NEW.INSTANCE )
--                 when matched then update 
--                    SET N_CREATED = NVL(N_CREATED,0) + 1
--                 when not matched then 
-- 	   	   INSERT (                
--                       RUNNUMBER,
--                       INSTANCE,
--                       HOSTNAME, 
--                       N_CREATED,
--                       START_WRITE_TIME,
--                       SETUPLABEL)
--                     VALUES (
--                       :NEW.RUNNUMBER,
--                       :NEW.INSTANCE,
--                       :NEW.HOSTNAME,
--                       1,
--                       :NEW.CTIME,
--                       :NEW.SETUPLABEL);
--  
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || ' 115-FILES_CREATED_AI: Done LOCK/INSERT SM_INSTANCES SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' SM_SUMMARY  FILE: '|| :NEW.FILENAME || '   <<');
-- 
-- 
--         ELSE
-- 
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || ' 123-FILES_CREATED_AI: preUpdate  SM_INSTANCES  <<');
-- 
--              UPDATE SM_INSTANCES
--                SET N_CREATED = NVL(N_CREATED,0) + 1
--                WHERE RUNNUMBER = :NEW.RUNNUMBER AND INSTANCE = :NEW.INSTANCE;
-- 
--              v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--              DBMS_OUTPUT.PUT_LINE ( v_etime || ' 124-FILES_CREATED_AI: postUpdate on  SM_INSTANCES  SQL%ROWCOUNT: ' || SQL%ROWCOUNT  || ' <<');
-- 
--         END IF;
-- 
--         v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--         DBMS_OUTPUT.PUT_LINE ( v_etime || ' 155-FILES_CREATED_AI: done  SM_INSTANCES FILE: '|| :NEW.FILENAME || ' -----------  <<');
-- 
-- 
--         EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_INJECTED so handle it with message
--              WHEN DUP_VAL_ON_INDEX THEN
--                 DBMS_OUTPUT.PUT_LINE ('**ERROR: FILES_CREATED_AI  DUP_VAL_ON_INDEX for SM_INSTANCES; FILE::' ||:NEW.FILENAME || '  << ');
--                 WHEN OTHERS THEN
--                    v_code := SQLCODE;
--                    v_errm := SUBSTR(SQLERRM, 1 , 64);
--                    v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--                   DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_CREATED_AI  for SM_INSTANCES: ' ||  v_errm || ' <<');
--                   DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_CREATED_AI  for SM_INSTANCES; FILE: ' || :NEW.FILENAME || '  << ');
-- 
--      END;
--      DBMS_OUTPUT.PUT_LINE ('-------FILES_CREATED Done :' ||:NEW.FILENAME || '-------------------' );
-- 
-- 
--    END IF;
-- 
-- END;
-- /


CREATE OR REPLACE TRIGGER FILES_DELETED_AI
AFTER INSERT ON FILES_DELETED
FOR EACH ROW
DECLARE 
v_producer VARCHAR(30);
v_stream VARCHAR(30);
v_instance NUMBER(5);
v_runnumber NUMBER(10);
v_code NUMBER;
v_errm VARCHAR2(64);
v_etime VARCHAR2(64);
BEGIN
     SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER into v_producer, v_stream, v_instance, v_runnumber FROM FILES_CREATED WHERE FILENAME = :NEW.FILENAME;
     IF v_producer = 'StorageManager' THEN
     	UPDATE SM_SUMMARY
        	SET S_DELETED = NVL(S_DELETED,0) + 1,
	    	LAST_UPDATE_TIME = sysdate,
                STOP_DELETE_TIME = sysdate
      	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
      	IF SQL%ROWCOUNT = 0 THEN
	  	NULL;
      	END IF;
        UPDATE SM_INSTANCES
                SET N_DELETED = NVL(N_DELETED,0) + 1,
                STOP_DELETE_TIME = sysdate
        WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
        IF SQL%ROWCOUNT = 0 THEN
                NULL;
        END IF;
     END IF;
     EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_DELETED so handle it with message
        WHEN OTHERS THEN
                v_code := SQLCODE;
                v_errm := SUBSTR(SQLERRM, 1 , 64);
                v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
                DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_DELETED_AI: ' ||  v_errm || ' <<');
                DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_DELETED_AI for FILE: ' || :NEW.FILENAME || '  << ');
END;
/

-- Migrated to a stored procedure
-- CREATE OR REPLACE TRIGGER FILES_INJECTED_AI
-- AFTER INSERT ON FILES_INJECTED
-- FOR EACH ROW
-- DECLARE 
-- v_producer VARCHAR(30);
-- v_stream VARCHAR(30);
-- v_instance NUMBER(5);
-- v_runnumber NUMBER(10);
-- v_code NUMBER;
-- v_errm VARCHAR2(64);
-- v_etime VARCHAR2(64);
-- BEGIN
--      SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER into v_producer, v_stream, v_instance, v_runnumber FROM FILES_CREATED WHERE FILENAME = :NEW.FILENAME;
--      IF v_producer = 'StorageManager' THEN
--        BEGIN
--          v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--          DBMS_OUTPUT.PUT_LINE ( v_etime || '  1-FILES_INJECTED: pre UPDATE  SM_SUMMARY    <<'); 
-- 
--        	 UPDATE SM_SUMMARY
--                  SET S_NEVENTS = NVL(S_NEVENTS,0) + NVL(:NEW.NEVENTS,0),
--             	 S_FILESIZE = NVL(S_FILESIZE,0) + NVL(:NEW.FILESIZE,0),
--             	 S_FILESIZE2D = NVL(S_FILESIZE2D,0) + NVL(:NEW.FILESIZE,0),
--             	 S_INJECTED = NVL(S_INJECTED,0) + 1,
--                  N_INSTANCE = (SELECT COUNT(DISTINCT INSTANCE) FROM FILES_CREATED WHERE RUNNUMBER = v_runnumber AND STREAM = v_stream),
-- 	    	 STOP_WRITE_TIME = GREATEST(:NEW.ITIME, NVL(STOP_WRITE_TIME, :NEW.ITIME)),
-- 	    	 HLTKEY = NVL(HLTKEY, :NEW.COMMENT_STR),
--             	 LAST_UPDATE_TIME = sysdate
--       	 WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
--      	 IF SQL%ROWCOUNT = 0 THEN
--          	NULL;
--      	 END IF;
--          EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_INJECTED so handle it with message
--            WHEN OTHERS THEN
--                 v_code := SQLCODE;
--                 v_errm := SUBSTR(SQLERRM, 1 , 64);
--                  v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_INJECTED_AI SM_SUMMARY: ' ||  v_errm || ' <<');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_INJECTED_AI SM_SUMMARY: for FILE: ' || :NEW.FILENAME || '  << '); 
--         END;
-- 
--         BEGIN
--           v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--           DBMS_OUTPUT.PUT_LINE (v_etime || '  3-FILES_INJECTED:  pre-UPDATE INSTANCES   <<');
-- 
--           UPDATE SM_INSTANCES
--                   SET N_INJECTED = NVL(N_INJECTED,0) + 1,
--                   LAST_WRITE_TIME = GREATEST(:NEW.ITIME, NVL(LAST_WRITE_TIME, :NEW.ITIME))
--           WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
--           IF SQL%ROWCOUNT = 0 THEN
--                   NULL;
--           END IF;
-- 
-- 
--           v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--           DBMS_OUTPUT.PUT_LINE (v_etime || '  5-FILES_INJECTED: ALL done!  FILE: '|| :NEW.FILENAME || ' -------   <<');
--           EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_INJECTED so handle it with message
--             WHEN OTHERS THEN
--                 v_code := SQLCODE;
--                 v_errm := SUBSTR(SQLERRM, 1 , 64);
--                 v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_INJECTED_AI: ' ||  v_errm || ' <<');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_INJECTED_AI for FILE: ' || :NEW.FILENAME || '  << ');
--         END;
--      END IF;
-- END;
-- /

-- Migrated to a stored procedure
-- CREATE OR REPLACE TRIGGER FILES_TRANS_CHECKED_AI
-- AFTER INSERT ON FILES_TRANS_CHECKED
-- FOR EACH ROW
-- DECLARE 
-- v_producer VARCHAR(30);
-- v_stream VARCHAR(30);
-- v_instance NUMBER(5);
-- v_runnumber NUMBER(10);
-- v_code NUMBER;
-- v_errm VARCHAR2(64);
-- v_etime VARCHAR2(64);
-- BEGIN
--      SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER into v_producer, v_stream, v_instance, v_runnumber FROM FILES_CREATED WHERE FILENAME = :NEW.FILENAME;
--      IF v_producer = 'StorageManager' THEN
--      	UPDATE SM_SUMMARY
--         	SET S_CHECKED = NVL(S_CHECKED,0) + 1,
--             	START_REPACK_TIME = LEAST(:NEW.ITIME, NVL(START_REPACK_TIME,:NEW.ITIME)),
-- 	    	LAST_UPDATE_TIME = sysdate       
--       	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
--      	IF SQL%ROWCOUNT = 0 THEN
--          	NULL;
--      	END IF;
--         UPDATE SM_INSTANCES
--                 SET N_CHECKED = NVL(N_CHECKED,0) + 1
--         WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
--         IF SQL%ROWCOUNT = 0 THEN
--                 NULL;
--         END IF;
--      END IF;
--      EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_TRANS_CHECKED so handle it with message
--         WHEN OTHERS THEN
--                 v_code := SQLCODE;
--                 v_errm := SUBSTR(SQLERRM, 1 , 64);
--                 v_etime := to_char(sysdate, 'Dy Mon DD HH24:MI:SS YYYY');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_CHECKED_AI: ' ||  v_errm || ' <<');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_CHECKED_AI for FILE: ' || :NEW.FILENAME || '  << ');
-- END;
-- /

-- Migrated to a stored procedure
-- CREATE OR REPLACE TRIGGER FILES_TRANS_COPIED_AI
-- AFTER INSERT ON FILES_TRANS_COPIED
-- FOR EACH ROW
-- DECLARE
-- v_producer VARCHAR(30);
-- v_stream VARCHAR(30);
-- v_instance NUMBER(5);
-- v_runnumber NUMBER(10);
-- v_code NUMBER;
-- v_errm VARCHAR2(64);
-- v_etime VARCHAR2(64);
-- BEGIN
--      SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER into v_producer, v_stream, v_instance, v_runnumber FROM FILES_CREATED WHERE FILENAME = :NEW.FILENAME;
--      IF v_producer = 'StorageManager' THEN
--      	UPDATE SM_SUMMARY
--         	SET S_COPIED = NVL(S_COPIED,0) + 1,
-- 	    	STOP_TRANS_TIME = GREATEST(:NEW.ITIME, NVL(STOP_TRANS_TIME, :NEW.ITIME)),
--             	S_FILESIZE2T0 = NVL(S_FILESIZE2T0,0) + 
-- 			NVL((SELECT FILESIZE from FILES_INJECTED where FILENAME = :NEW.FILENAME),0),
--             	LAST_UPDATE_TIME = sysdate
--       	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
--      	IF SQL%ROWCOUNT = 0 THEN
--          	NULL;
--      	END IF;
--         UPDATE SM_INSTANCES
--                 SET N_COPIED = NVL(N_COPIED,0) + 1
--         WHERE RUNNUMBER = v_runnumber AND INSTANCE=v_instance;
--         IF SQL%ROWCOUNT = 0 THEN
--                 NULL;
--         END IF;
--      END IF;
--      EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_TRANS_COPIED so handle it with message
--         WHEN OTHERS THEN
--                 v_code := SQLCODE;
--                 v_errm := SUBSTR(SQLERRM, 1 , 64);
--                 v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_COPIED_AI: ' ||  v_errm || ' <<');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_COPIED_AI for FILE: ' || :NEW.FILENAME || '  << ');
-- END;
-- /

-- Migrated to a stored procedure
-- CREATE OR REPLACE TRIGGER FILES_TRANS_NEW_AI
-- AFTER INSERT ON FILES_TRANS_NEW
-- FOR EACH ROW
-- DECLARE 
-- v_producer VARCHAR(30);
-- v_stream VARCHAR(30);
-- v_instance NUMBER(5);
-- v_runnumber NUMBER(10);
-- v_code NUMBER;
-- v_errm VARCHAR2(64);
-- v_etime VARCHAR2(64);
-- BEGIN
--      SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER into v_producer, v_stream, v_instance, v_runnumber FROM FILES_CREATED WHERE FILENAME = :NEW.FILENAME;
--      IF v_producer = 'StorageManager' THEN
--      	UPDATE SM_SUMMARY
--         	SET S_NEW = NVL(S_NEW,0) + 1,
-- 	    	START_TRANS_TIME =  LEAST(:NEW.ITIME, NVL(START_TRANS_TIME,:NEW.ITIME)),
--             	LAST_UPDATE_TIME = sysdate
--       	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
--      	IF SQL%ROWCOUNT = 0 THEN
--          	NULL;
--      	END IF;
--         UPDATE SM_INSTANCES
--                 SET N_NEW = NVL(N_NEW,0) + 1
--         WHERE RUNNUMBER = v_runnumber AND INSTANCE = v_instance;
--         IF SQL%ROWCOUNT = 0 THEN
--                 NULL;
--         END IF;
--      END IF;
--      EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_TRANS_NEW so handle it with message
--         WHEN OTHERS THEN
--                 v_code := SQLCODE;
--                 v_errm := SUBSTR(SQLERRM, 1 , 64);
--                 v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_NEW_AI: ' ||  v_errm || ' <<');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_NEW_AI for FILE: ' || :NEW.FILENAME || '  << ');
--  END;
-- /

-- Migrated to a stored procedure
-- CREATE OR REPLACE TRIGGER FILES_TRANS_REPACKED_AI
-- AFTER INSERT ON FILES_TRANS_REPACKED
-- FOR EACH ROW
-- DECLARE 
-- v_producer VARCHAR(30);
-- v_stream VARCHAR(30);
-- v_instance NUMBER(5);
-- v_runnumber NUMBER(10);
-- v_code NUMBER;
-- v_errm VARCHAR2(64);
-- v_etime VARCHAR2(64);
-- BEGIN
--      SELECT PRODUCER, STREAM, INSTANCE, RUNNUMBER into v_producer, v_stream, v_instance, v_runnumber FROM FILES_CREATED WHERE FILENAME = :NEW.FILENAME;
--      IF v_producer = 'StorageManager' THEN
--      	UPDATE SM_SUMMARY
--         	SET S_REPACKED = NVL(S_REPACKED,0) + 1,
-- 	    	STOP_REPACK_TIME = :NEW.ITIME,
-- 	    	LAST_UPDATE_TIME = sysdate
--       	WHERE RUNNUMBER = v_runnumber AND STREAM=v_stream;
--      	IF SQL%ROWCOUNT = 0 THEN
--         	 NULL;
--      	END IF;
--         UPDATE SM_INSTANCES
--                 SET N_REPACKED = NVL(N_REPACKED,0) + 1
--         WHERE RUNNUMBER = v_runnumber AND INSTANCE=v_instance;
--         IF SQL%ROWCOUNT = 0 THEN
--                 NULL;
--         END IF;
--      END IF;
--      EXCEPTION  --what if error?? do NOT want ERROR to propagate to FILES_TRANS_REPACKED so handle it with message
--         WHEN OTHERS THEN
--                 v_code := SQLCODE;
--                 v_errm := SUBSTR(SQLERRM, 1 , 64);
--                 v_etime := to_char(systimestamp, 'Dy Mon DD HH24:MI:SS.FF5  YYYY');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_REPACKED_AI: ' ||  v_errm || ' <<');
--                 DBMS_OUTPUT.PUT_LINE ('##   ' || v_etime ||' ERROR: FILES_TRANS_REPACKED_AI for FILE: ' || :NEW.FILENAME || '  << ');
-- END;
-- /
