/* $id$ 
 * 
 * Procedure to validate an IoV to be inserted and update a previous
 * IoV so that there are no overlaps.
 * 
 * GO: september 2010
 * This new procedure allows multiple IOVs with the same start date
 * IOVs have a mask based on which one can assign a given IOV to a given table
 * To test the script you have to run generate_iovs.pl before and send its
 * output to ttt.sql
 * ./generate_iovs.sql > ttt.sql
 * then run this script and check results using report.txt
 * 
 */

WHENEVER SQLERROR EXIT

/* 
   This first part of the script is intended to avoi running it on the
   wrong database
*/

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

/* Ok: we are running on the right database */

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
  
  sql_str    VARCHAR(1000);
  tn         VARCHAR(25);
  last_start DATE;
  last_iov   NUMBER;
  new_iov    NUMBER;
  last_mask  NUMBER;
  rows       NUMBER;
  i          NUMBER;

  BEGIN
    dbms_output.enable;
    -- Ensure IoV time has positive duration
    IF new_end <= new_start THEN
       dbms_output.put_line('WARNING: ' || new_start || ' >= ' || new_end);
       raise_application_error(-20000, 'IOV must have ' || start_col || ' < ' 
                               || end_col);
    END IF;
    -- Look for records containing this mask
    sql_str := 'SELECT COUNT(IOV_ID) FROM ' || my_table || 
      ' WHERE BITAND(MASK, ' ||
	my_mask || ') > 0 AND TAG_ID = :t AND ' || end_col || 
	' >= TO_DATE(''31-12-9999 23:59:59'', ''DD-MM-YYYY HH24:MI:SS'')';
    EXECUTE IMMEDIATE sql_str INTO rows USING new_tag_id;
    dbms_output.put_line('Found ' || rows || ' rows with good mask');
    -- Case select
    IF rows = 0 THEN
       -- do nothing, just insert the row
       return;
    ELSE
       -- IOV_ID found with the same bits: update them
       FOR i IN 1..rows LOOP
          sql_str := 'SELECT IOV_ID, MASK FROM ' || my_table || 
             ' WHERE BITAND(MASK, ' ||
	     my_mask || ') > 0 AND TAG_ID = :t AND ' || end_col || 
	     ' >= TO_DATE(''31-12-9999 23:59:59'', ''DD-MM-YYYY HH24:MI:SS'')' ||
             ' AND ROWNUM = 1';
	  dbms_output.put_line(sql_str);
          EXECUTE IMMEDIATE sql_str INTO last_iov, last_mask USING new_tag_id;
	  dbms_output.put_line('  IOV: ' || last_iov);
	  dbms_output.put_line(' MASK: ' || last_mask);
          sql_str := 'UPDATE ' || my_table || ' SET ' || end_col || 
	     ' = :new_start WHERE IOV_ID = :last_iov';
          EXECUTE IMMEDIATE sql_str USING new_start, last_iov;
       END LOOP;
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


CREATE SEQUENCE TEST_IOV_SQ START WITH 1 NOCACHE;

/* simulate measurements */

@ttt

SELECT * FROM TEST_IOV;
SELECT * FROM TEST_IOV JOIN TEST_A ON TEST_IOV.IOV_ID = TEST_A.IOV_ID 
	WHERE BITAND(MASK, 1) = 1 ORDER BY SINCE ASC;
SELECT * FROM TEST_IOV JOIN TEST_B ON TEST_IOV.IOV_ID = TEST_B.IOV_ID 
	WHERE BITAND(MASK, 2) = 2 ORDER BY SINCE ASC;
SELECT * FROM TEST_IOV JOIN TEST_C ON TEST_IOV.IOV_ID = TEST_C.IOV_ID 
	WHERE BITAND(MASK, 4) = 4 ORDER BY SINCE ASC;
