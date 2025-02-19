set serveroutput on size 100000 format wrapped;

DECLARE

TYPE beams_ary IS VARRAY (2) OF VARCHAR2(2);
beams beams_ary;
beam VARCHAR2(2);
create_sql VARCHAR2(10000);
tbl VARCHAR2(32);
var VARCHAR2(32);
field VARCHAR2(255);

BEGIN
  beams := beams_ary('H4', 'H2');
  
  FOR i IN 1..2
  LOOP
    beam := beams(i);
    tbl := 'RUN_' || beam || '_BEAM_DAT';
    create_sql := 'CREATE TABLE ' || tbl || ' (iov_id NUMBER(10), ';
    dbms_output.put_line(create_sql);

    FOR result IN (SELECT variable_name, datatype FROM beam_source WHERE beamline=beam ORDER BY variable_name)
    LOOP
      var := result.variable_name;

      IF result.datatype = 'NUMERIC' THEN
        field := '"' || var || '"' || ' NUMBER';
      ELSIF result.datatype = 'TEXTUAL' THEN
        field := '"' || var || '"' || ' VARCHAR2(100)';
      ELSE raise_application_error(-20000, 'UNKNOWN DATATYPE');
      END IF;

      dbms_output.put_line(field);
      create_sql := create_sql || field || ',';
    END LOOP;

    create_sql := trim(TRAILING ',' FROM create_sql) || ')';

    EXECUTE IMMEDIATE create_sql;
    EXECUTE IMMEDIATE 'ALTER TABLE ' || tbl || ' ADD CONSTRAINT ' 
                      || tbl || '_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id)';
  END LOOP;
END;
/
