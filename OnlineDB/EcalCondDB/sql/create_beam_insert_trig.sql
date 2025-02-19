CREATE OR REPLACE TRIGGER beam_insert_tg
AFTER INSERT ON run_iov
REFERENCING NEW AS newrun
FOR EACH ROW

DECLARE
  loc VARCHAR2(100);
  tbl VARCHAR2(32);
  sql_str VARCHAR2(4000);
BEGIN
  SELECT location INTO loc 
    FROM run_tag rtag JOIN location_def ldef ON rtag.location_id = ldef.def_id 
   WHERE rtag.tag_id = :newrun.tag_id;

  IF    loc = 'H4B' THEN 
    tbl := 'RUN_H4_BEAM_DAT';
  ELSIF loc = 'H2' THEN 
    tbl := 'RUN_H2_BEAM_DAT';
  ELSE return;
  END IF;

  sql_str := 'INSERT INTO ' || tbl || '(iov_id) VALUES (:iov_id)';
  EXECUTE IMMEDIATE sql_str USING :newrun.iov_id;

  FOR col IN (SELECT bs.variable_name, bs.datatype, bs.value, bs.string_value FROM user_tab_cols ucols, beam_source bs 
               WHERE ucols.column_name = bs.variable_name 
                 AND ucols.table_name = tbl 
                 AND ucols.column_name != 'IOV_ID') 
  LOOP
    sql_str := 'UPDATE ' || tbl || ' SET "' || col.variable_name || '" = :val WHERE iov_id = :iov_id';
    IF col.datatype = 'NUMERIC' THEN
      EXECUTE IMMEDIATE sql_str USING col.value, :newrun.iov_id;
    ELSIF col.datatype = 'TEXTUAL' THEN
      EXECUTE IMMEDIATE sql_str USING col.string_value, :newrun.iov_id;
    ELSE raise_application_error(-20000, 'UNKNOWN DATATYPE');
    END IF;

  END LOOP;
END;
/