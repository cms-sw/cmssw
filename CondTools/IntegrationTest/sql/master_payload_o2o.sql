/*
 * master_payload_o2o.sql
 *
 * PL/SQL procedure to execute O2O procedure for object_name and do all general bookkeeping and logging.
 * Parameters:  object_name:  Name of the object/prefix of the procedure to execute O2O on
 *              
 */

CREATE OR REPLACE PROCEDURE master_payload_o2o (
  object_name IN VARCHAR2
)
AS

top_level_table VARCHAR2(32);
db_link VARCHAR2(32);
table_link VARCHAR2(65);
start_time TIMESTAMP;
plsql_block VARCHAR2(256);
last_id NUMBER(10);
cnt1 NUMBER(10);
cnt2 NUMBER(10);

BEGIN
  -- Check that <object_name>_payload_o2o() exists

  -- Get the top level table
  SELECT top_level_table, db_link INTO top_level_table, db_link 
    FROM O2O_SETUP WHERE object_name = object_name;
  
  -- Check the setup variables

  table_link := top_level_table || '@' || db_link;

  -- Get the count and ID variables
  EXECUTE IMMEDIATE 'SELECT count(*) FROM ' || table_link INTO cnt1;
  EXECUTE IMMEDIATE 'SELECT max(IOV_VALUE_ID) FROM ' || table_link INTO last_id;

  IF last_id IS NULL THEN
    -- It is the first transfer, ensure ALL ids are transferred
    -- Use -1 because all IOV_VALUE_ID are > 0
    last_id := -1;
  END IF;

  -- Get the start time
  SELECT systimestamp INTO start_time FROM dual;

  -- Execute transfer
  plsql_block := 'BEGIN ' || object_name || '_payload_o2o@' || db_link ||'(:last_id); END;';
  EXECUTE IMMEDIATE plsql_block USING last_id;

  -- Get the count and ID variables again
  EXECUTE IMMEDIATE 'SELECT count(*) FROM ' || table_link INTO cnt2;
  EXECUTE IMMEDIATE 'SELECT max(IOV_VALUE_ID) FROM ' || table_link INTO last_id;


  -- Write into O2O_LOG
  INSERT INTO o2o_log (object_name, last_id, num_transfered, transfer_start, transfer_duration)
       VALUES (object_name, last_id, cnt2 - cnt1, start_time, start_time - systimestamp);

  COMMIT;

END;
/
show errors;
