/* $id$ 
 * 
 * Procedure to validate an IoV to be inserted and update a previous
 * IoV so that there are no overlaps.
 * 
 * GO: september 2010
 * This new procedure allows multiple IOVs with the same start date
 * IOVs have a mask based on which one can assign a given IOV to a given table
 *	
 */

WHENEVER SQLERROR EXIT

CREATE OR REPLACE PROCEDURE TESTDB IS
   gname VARCHAR2(64);
BEGIN
   SELECT GLOBAL_NAME INTO gname FROM GLOBAL_NAME;
   IF gname = 'INT2R.CERN.CH' THEN
      -- do nothing
      return;
   ELSE
      RAISE_APPLICATION_ERROR(-20001, 'Wrong DB name: ' || gname);
   END IF;
END;
/

EXEC TESTDB;

CREATE OR REPLACE PROCEDURE update_iov_dates_test
( my_table IN VARCHAR2,
  my_mask IN NUMBER,
  my_sequence IN VARCHAR2,
  my_dattable IN VARCHAR2,  
  start_col IN VARCHAR2,
  end_col IN VARCHAR2,
  new_start IN DATE,
  new_end IN OUT DATE,
  new_tag_id IN NUMBER ) IS
  
  sql_str VARCHAR(1000);
  future_start DATE;
  dat_table_id NUMBER;
  table_name VARCHAR(25);
  tN VARCHAR(25);
  last_start DATE;
  last_end DATE;
  last_iov NUMBER;
  new_iov NUMBER;
  last_mask NUMBER;
  rows NUMBER;
  I NUMBER;

  BEGIN
    dbms_output.enable;
    -- Ensure IoV time has positive duration
    IF new_end <= new_start THEN
       raise_application_error(-20000, 'IOV must have ' || start_col || ' < ' 
                               || end_col);
    END IF;
    -- Look for records containing this mask
    sql_str := 'SELECT IOV_ID FROM ' || my_table || 
      ' WHERE BITAND(MASK, ' ||
	my_mask || ') > 0 AND TAG_ID = :t AND ' || end_col || 
	' >= TO_DATE(''31-12-9999 23:59:59'', ''DD-MM-YYYY HH24:MI:SS'')';
    EXECUTE IMMEDIATE sql_str INTO last_iov USING new_tag_id;
    IF last_iov IS NOT NULL THEN 
       -- record found
       sql_str := 'SELECT MASK FROM ' || my_table || 
	  ' WHERE IOV_ID = :last_iov';
       EXECUTE IMMEDIATE sql_str INTO last_mask USING last_iov;
       dbms_output.put_line('LAST_IOV is ' || last_iov);
       dbms_output.put_line('MASKS: ' || my_mask || ' - ' || last_mask);
       IF my_mask >= last_mask THEN
          -- mask found
          sql_str := 'UPDATE ' || my_table || ' SET ' || end_col || 
	     ' = :new_start WHERE IOV_ID = :last_iov';
          EXECUTE IMMEDIATE sql_str USING new_start, last_iov;
       ELSE
          -- a new mask should be created: get the IOV_ID of the last
          -- measurement
          sql_str := 'SELECT ' || start_col || ' FROM ' || my_table || 
	     ' WHERE IOV_ID = :last_iov';
          EXECUTE IMMEDIATE sql_str INTO last_start USING last_iov;
          dbms_output.put_line('SPECIAL: changing mask from ' || last_mask);
          last_mask := last_mask - my_mask;
          dbms_output.put_line('SPECIAL:                 to ' || last_mask);
	  -- update the mask of the last measurement
	  sql_str := 'UPDATE ' || my_table || ' SET MASK = :last_mask WHERE '
	     || 'IOV_ID = :last_iov';
          EXECUTE IMMEDIATE sql_str USING last_mask, last_iov;
          -- insert a record with the given mask
	  sql_str := 'SELECT ' || my_sequence || '.NextVal FROM DUAL';
          EXECUTE IMMEDIATE sql_str INTO new_iov;
	  sql_str := 'INSERT INTO ' || my_table || ' VALUES (:new_iov, ' ||
	     ':my_mask, :new_tag_id, :last_start, :new_start)';
	  EXECUTE IMMEDIATE sql_str USING new_iov, my_mask, new_tag_id,
	      last_start, new_start;
          -- get the affected tables
	  sql_str := 'SELECT COUNT(*) FROM ' || my_dattable || ' WHERE ' ||
	      'BITAND(ID, ' || my_mask || ') > 0';
          EXECUTE IMMEDIATE sql_str INTO rows;
	  dbms_output.put_line('Tables to update: ' || rows);
          FOR i IN 1..rows LOOP
  	     sql_str := 'SELECT TABLE_NAME FROM ' || my_dattable || 
	       ' WHERE BITAND(ID, ' || my_mask || ') > 0 AND ROWNUM = :i';
             EXECUTE IMMEDIATE sql_str INTO tn USING i;
             sql_str := 'UPDATE ' || tn || 
	        ' SET IOV_ID = :new_iov WHERE IOV_ID = :last_iov';
	     EXECUTE IMMEDIATE sql_str USING new_iov, last_iov;
          END LOOP;	
       END IF;
    END IF;
    EXCEPTION
    WHEN NO_DATA_FOUND THEN
       dbms_output.put_line('NO DATA FOUND');
  END;
/

show errors;

SET ECHO OFF
SET HEADING OFF 
SPOOL tmp.sql
SELECT 'DROP TABLE ' || TABLE_NAME || ';' FROM USER_TABLES WHERE 
	TABLE_NAME LIKE 'TEST%';
SELECT 'DROP SEQUENCE ' || SEQUENCE_NAME || ';' FROM USER_SEQUENCES WHERE
	SEQUENCE_NAME LIKE 'TEST%';
SPOOL OFF
@tmp;

SET HEADING ON

CREATE TABLE TEST_IOV (
  IOV_ID NUMBER,
  MASK NUMBER,
  TAG_ID NUMBER,
  SINCE DATE,
  TILL DATE
);

CREATE TABLE TEST_DATTABLE_ID (
  ID NUMBER,
  TABLE_NAME VARCHAR2(25),
  DESCRIPTION VARCHAR2(255)
);

INSERT INTO TEST_DATTABLE_ID VALUES (1, 'TEST_A', '');
INSERT INTO TEST_DATTABLE_ID VALUES (2, 'TEST_B', '');
INSERT INTO TEST_DATTABLE_ID VALUES (4, 'TEST_C', '');

CREATE TABLE TEST_A (
  IOV_ID NUMBER,
  DATA NUMBER
);

CREATE TABLE TEST_B (
  IOV_ID NUMBER,
  DATA NUMBER
);

CREATE TABLE TEST_C (
  IOV_ID NUMBER,
  DATA NUMBER
);

CREATE OR REPLACE TRIGGER TEST_IOV_TG
        BEFORE INSERT ON TEST_IOV
        REFERENCING NEW AS newiov
        FOR EACH ROW
        CALL update_iov_dates_test('TEST_IOV', :newiov.mask,
        'TEST_IOV_SQ', 'TEST_DATTABLE_ID', 'SINCE', 'TILL',
        :newiov.since,
        :newiov.till, :newiov.tag_id)
/

show errors;

alter session set NLS_DATE_FORMAT='DD-MM-YYYY HH24:MI:SS';
set linesize 110
SET SERVEROUTPUT ON

/* simulate measurements
   1 made on 1, 2, 3, 4, 6 feb
   2 made on 4, 6, 7 feb
   4 made on 5 feb
*/

CREATE SEQUENCE TEST_IOV_SQ START WITH 1 NOCACHE;
insert into test_iov values(TEST_IOV_SQ.NextVal, 1, 1, 
	to_date('01-01-2010 00:00:00', 'DD-MM-YYYY HH24:MI:SS'), 
	to_date('31-12-9999 23:59:59', 'DD-MM-YYYY HH24:MI:SS'));
INSERT INTO TEST_A VALUES (1, 1);
insert into test_iov values(TEST_IOV_SQ.NextVal, 1, 1, 
	to_date('02-01-2010 00:00:00', 'DD-MM-YYYY HH24:MI:SS'), 
	to_date('31-12-9999 23:59:59', 'DD-MM-YYYY HH24:MI:SS'));
INSERT INTO TEST_A VALUES (2, 2);
insert into test_iov values(TEST_IOV_SQ.NextVal, 1, 1, 
	to_date('03-01-2010 00:00:00', 'DD-MM-YYYY HH24:MI:SS'), 
	to_date('31-12-9999 23:59:59', 'DD-MM-YYYY HH24:MI:SS'));
INSERT INTO TEST_A VALUES (3, 3);
insert into test_iov values(TEST_IOV_SQ.NextVal, 3, 1, 
	to_date('04-01-2010 00:00:00', 'DD-MM-YYYY HH24:MI:SS'), 
	to_date('31-12-9999 23:59:59', 'DD-MM-YYYY HH24:MI:SS'));
INSERT INTO TEST_A VALUES (4, 4);
INSERT INTO TEST_B VALUES (4, 1);
insert into test_iov values(TEST_IOV_SQ.NextVal, 4, 1, 
	to_date('05-01-2010 00:00:00', 'DD-MM-YYYY HH24:MI:SS'), 
	to_date('31-12-9999 23:59:59', 'DD-MM-YYYY HH24:MI:SS'));
INSERT INTO TEST_C VALUES (5, 1);
insert into test_iov values(TEST_IOV_SQ.NextVal, 3, 1, 
	to_date('06-01-2010 00:00:00', 'DD-MM-YYYY HH24:MI:SS'), 
	to_date('31-12-9999 23:59:59', 'DD-MM-YYYY HH24:MI:SS'));
INSERT INTO TEST_A VALUES (6, 5);
INSERT INTO TEST_B VALUES (7, 2);
insert into test_iov values(TEST_IOV_SQ.NextVal, 2, 1, 
	to_date('07-01-2010 00:00:00', 'DD-MM-YYYY HH24:MI:SS'), 
	to_date('31-12-9999 23:59:59', 'DD-MM-YYYY HH24:MI:SS'));
INSERT INTO TEST_B VALUES (8, 3);

SELECT * FROM TEST_IOV;
SELECT * FROM TEST_IOV JOIN TEST_A ON TEST_IOV.IOV_ID = TEST_A.IOV_ID 
	WHERE BITAND(MASK, 1) = 1 ORDER BY SINCE ASC;
SELECT * FROM TEST_IOV JOIN TEST_B ON TEST_IOV.IOV_ID = TEST_B.IOV_ID 
	WHERE BITAND(MASK, 2) = 2 ORDER BY SINCE ASC;
SELECT * FROM TEST_IOV JOIN TEST_C ON TEST_IOV.IOV_ID = TEST_C.IOV_ID 
	WHERE BITAND(MASK, 4) = 4 ORDER BY SINCE ASC;
