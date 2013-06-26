set serveroutput on;

CREATE OR REPLACE PROCEDURE poll_db 
AS

tbl varchar2(32);
ts timestamp;
cnt number(10);

BEGIN
  SELECT systimestamp INTO ts FROM dual;
  dbms_output.put_line('Time:  ' || ts);
  FOR result IN (SELECT table_name FROM user_tables ORDER BY table_name)
  LOOP
    EXECUTE IMMEDIATE 'SELECT ''' || result.table_name || ''', count(*) FROM ' || result.table_name INTO tbl, cnt;
    dbms_output.put_line(tbl || ' ' || cnt || ' rows');
  END LOOP;
END;
/
show errors;
