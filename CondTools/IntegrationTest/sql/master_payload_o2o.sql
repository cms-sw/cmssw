/*
 * master_payload_o2o.sql
 *
 * PL/SQL procedure to execute O2O procedure for object_name and do all general bookkeeping and logging.
 * Parameters:  object_name:  Name of the object/prefix of the procedure to execute O2O on
 *              
 */

CREATE OR REPLACE PROCEDURE master_payload_o2o (
  i_object_name IN VARCHAR2
)
AS

schema VARCHAR2(32) := NULL;
top_level_table VARCHAR2(32) := NULL;
schema_table VARCHAR(65) := NULL;
start_time TIMESTAMP := NULL;
plsql_block VARCHAR2(256) := NULL;
last_id NUMBER(10) := NULL;
cnt1 NUMBER(10) := 0;
cnt2 NUMBER(10) := 0;

BEGIN
  -- Check that <object_name>_payload_o2o() exists

  -- Get the top level table
  BEGIN
    SELECT schema, top_level_table INTO schema, top_level_table 
      FROM O2O_SETUP WHERE object_name = i_object_name;
  EXCEPTION
    WHEN NO_DATA_FOUND THEN
      raise_application_error(-20101, 'Object ' || i_object_name || ' not found in O2O_SETUP table');
  END;

  schema_table := schema || '.' || top_level_table;

  -- Get the count and ID variables
  BEGIN
    EXECUTE IMMEDIATE 'SELECT count(*) FROM ' || schema_table INTO cnt1;
    EXECUTE IMMEDIATE 'SELECT max(IOV_VALUE_ID) FROM ' || schema_table INTO last_id;
  EXCEPTION
    WHEN OTHERS THEN /* table not found */
      raise_application_error(-20102, 'Cannot read top-level-table ' || schema_table, TRUE);
  END;


  -- It is the first transfer, ensure ALL ids are transferred
  -- Use -1 because all IOV_VALUE_ID are > 0
  IF last_id IS NULL THEN
    last_id := -1;
  END IF;

  -- Get the start time
  SELECT systimestamp INTO start_time FROM dual;

  -- Execute transfer
  BEGIN
    plsql_block := 'BEGIN ' || schema || '.' || i_object_name || '_payload_o2o(:last_id); END;';
    EXECUTE IMMEDIATE plsql_block USING last_id;
  EXCEPTION
    WHEN OTHERS THEN /* procedure not found */
      raise_application_error(-20103, 'Cannot read ' || i_object_name || '_payload_o2o procedure', TRUE);
  END;

  -- Get the count and ID variables again
  EXECUTE IMMEDIATE 'SELECT count(*) FROM ' || schema_table INTO cnt2;
  EXECUTE IMMEDIATE 'SELECT max(IOV_VALUE_ID) FROM ' || schema_table INTO last_id;

  -- Write into O2O_LOG if there was a change
  IF cnt2 != cnt1 THEN
    INSERT INTO o2o_log (object_name, last_id, num_transferred, transfer_start, transfer_duration)
         VALUES (i_object_name, last_id, cnt2 - cnt1, start_time, start_time - systimestamp);
  END IF;
  COMMIT;

END;
/
show errors;
