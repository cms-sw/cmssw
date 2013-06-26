set linesize 1000;
set pagesize 50000;
set feedback off;
set serveroutput on size 1000000;

CREATE OR replace FUNCTION config_exists (cycle_id NUMBER := NULL)
  RETURN VARCHAR2
AS
  TYPE exists_cur_type IS REF CURSOR;
  exists_cur exists_cur_type;
  exists_bool CHAR(1);
  like_str VARCHAR2(100);
  like2_str VARCHAR2(100);
  result VARCHAR2(1000);
  CURSOR c1 IS SELECT table_name FROM user_tables WHERE (table_name LIKE like_str)  ORDER BY table_name ASC;
  CURSOR c2 IS SELECT table_name FROM user_tables WHERE (table_name LIKE like2_str) OR (table_name LIKE 'ECAL_SCAN_DAT') ORDER BY table_name ASC;
  t VARCHAR2(32);
  t2 VARCHAR2(32);
  t3 VARCHAR2(32);
  sql_str VARCHAR2(1000);
BEGIN

like_str := 'ECAL_%_CYCLE';
like2_str := 'ECAL_%_CONFIGURATION';
OPEN c1;
OPEN c2;
LOOP
  FETCH c2 INTO t2;
  FETCH c1 INTO t;
  EXIT WHEN c1%NOTFOUND;
  sql_str := 'SELECT ''1'' FROM ' || t || ' WHERE cycle_id = :cycle_id AND rownum = 1';
  OPEN exists_cur FOR sql_str USING cycle_id;
  FETCH exists_cur INTO exists_bool;
  IF exists_cur%FOUND THEN
	t3 := t2;
	if (t2  LIKE 'ECAL_SCAN_DAT' ) THEN
     	   t3 :='ECAL_SCAN_CONFIGURATION' ; 
	END IF;
    result := result || ',' || t3;
  END IF;
  CLOSE exists_cur;
END LOOP;
CLOSE c1;
CLOSE c2;

RETURN ltrim(result,',');

EXCEPTION
WHEN OTHERS THEN
raise_application_error(-20001,'EXCEPTION - '||SQLCODE||' -ERROR- '||SQLERRM);
END;
/
show errors;
;
