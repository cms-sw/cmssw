set linesize 1000;
set pagesize 50000;
set feedback off;
set serveroutput on size 1000000;

CREATE OR replace FUNCTION dat_exists (pre VARCHAR2 := NULL, iov_id NUMBER := NULL) 
  RETURN VARCHAR2
AS
  TYPE exists_cur_type IS REF CURSOR;
  exists_cur exists_cur_type;
  exists_bool CHAR(1);
  like_str VARCHAR2(100);
  result VARCHAR2(1000);
  CURSOR c1 IS SELECT table_name FROM user_tables WHERE table_name LIKE like_str ORDER BY table_name ASC;
  t VARCHAR2(32);
  sql_str VARCHAR2(1000);
BEGIN

like_str := pre || '_%_DAT';
OPEN c1;
LOOP
  FETCH c1 INTO t;
  EXIT WHEN c1%NOTFOUND;
  sql_str := 'SELECT ''1'' FROM ' || t || ' WHERE iov_id = :iov_id AND rownum = 1';
  OPEN exists_cur FOR sql_str USING iov_id;
  FETCH exists_cur INTO exists_bool;
  IF exists_cur%FOUND THEN
    result := result || ',' || t;
  END IF;
  CLOSE exists_cur;
END LOOP;
CLOSE c1;

RETURN ltrim(result,',');

EXCEPTION
WHEN OTHERS THEN
raise_application_error(-20001,'EXCEPTION - '||SQLCODE||' -ERROR- '||SQLERRM);
END;
/
show errors;
;

/* Executes the exists program */
SELECT iov_id, dat_exists('MON', iov_id) FROM mon_run_iov WHERE rownum < 100;
SELECT iov_id, dat_exists('DCU', iov_id) FROM dcu_iov WHERE rownum < 100;
