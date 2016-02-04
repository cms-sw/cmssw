/* $id$ 
 * 
 * Procedure to validate an IoV to be inserted and update a previous
 * IoV so that there are no overlaps.
 * 
 * GO: september 2010
 * This new procedure allows multiple IOVs with the same start date
 * IOVs have a mask based on which one can assign a given IOV to a given table
 *
 * To test the script you have to run generate_iovs.pl before and send its
 * output to ttt.sql
 * ./generate_iovs.sql > ttt.sql
 * then run this script and check results using report.txt
 * 
 * TODO: MAKE IT INDEPENDENT ON TRIGGER NAME
 *       FINISH
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

/* Ok: we are running on the right database */

CREATE OR REPLACE FUNCTION bitnot (x IN NUMBER) RETURN NUMBER AS
BEGIN
  return (-1-x);
END;
/

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
  tnc        NUMBER;
  last_start DATE;
  last_iov   NUMBER;
  new_iov    NUMBER;
  zero_iov   NUMBER;
  last_mask  NUMBER;
  new_mask   NUMBER;
  my_rows    NUMBER;
  i          NUMBER;
  j          NUMBER;

  BEGIN
    dbms_output.enable(100000);
    -- Ensure IoV time has positive duration
    IF new_end <= new_start THEN
       raise_application_error(-20000, 'IOV must have ' || start_col || ' < ' 
                               || end_col);
    END IF;
    -- Look for records containing this mask
    sql_str := 'SELECT COUNT(IOV_ID) FROM ' || my_table || 
      ' WHERE BITAND(MASK, :my_mask) > 0 AND TAG_ID = :t AND ' || end_col || 
	' >= TO_DATE(''31-12-9999 23:59:59'', ''DD-MM-YYYY HH24:MI:SS'')';
    EXECUTE IMMEDIATE sql_str INTO my_rows USING my_mask, new_tag_id;
    IF my_mask != 0 THEN
       dbms_output.put_line('----------------------------------------');
    END IF;
    dbms_output.put_line(sql_str);
    dbms_output.put_line('Searching for mask ' || my_mask || ' tag ' || new_tag_id);
    dbms_output.put_line('Found ' || my_rows || ' rows with good mask');
    -- Case select
    IF my_rows = 0 THEN
       -- do nothing, just insert the row
       dbms_output.put_line('Inserting row with mask ' || my_mask || ' and tag ' || new_tag_id);	
       return;
    ELSE
       -- IOV_ID found with the same bits: update them
       FOR i IN 1..my_rows LOOP
          -- look for IOV's with the same bits on
          sql_str := 'SELECT IOV_ID, MASK, SINCE FROM ' || my_table || 
             ' WHERE BITAND(MASK, ' ||
	     my_mask || ') > 0 AND TAG_ID = :t AND ' || end_col || 
	     ' >= TO_DATE(''31-12-9999 23:59:59'', ''DD-MM-YYYY HH24:MI:SS'')' ||
             ' AND ROWNUM = 1';
	  dbms_output.put_line(sql_str);
          EXECUTE IMMEDIATE sql_str INTO last_iov, last_mask, last_start USING new_tag_id;
          dbms_output.put_line('Required insertion of data with mask: ' || my_mask);
          dbms_output.put_line('Found  data with mask               : ' || last_mask 
             || ' and IOV ' || last_iov);
          dbms_output.put_line('       and start date on ' || last_start);
          -- update the mask of those measurements not yet redone
          new_mask := BITAND(last_mask, BITNOT(my_mask));
          IF new_mask > 0 THEN 
             sql_str := 'UPDATE ' || my_table || ' SET MASK = BITAND(' ||
                ':last_mask, BITNOT(:my_mask)) WHERE IOV_ID = :last_iov AND BITAND( ' 
                || ':last_mask , BITNOT(:my_mask)) > 0';
             dbms_output.put_line(sql_str);
             EXECUTE IMMEDIATE sql_str USING last_mask, my_mask, last_iov, last_mask,
	        my_mask;	
             -- insert new record and update related tables (use mask 0 to avoid
             -- retriggering this procedure)
	     last_mask := BITAND(last_mask, my_mask);
             sql_str := 'INSERT INTO ' || my_table || ' VALUES (' ||
	        my_sequence || '.NextVal, 0, :tag_id, :since, :till)';
	     dbms_output.put_line('INSERTing new record with mask ' || last_mask || 
                ' and since, till = ' || last_start || ', ' || new_start);
             EXECUTE IMMEDIATE sql_str USING new_tag_id, last_start, new_start;
	     -- get the last inserted id
             sql_str := 'SELECT IOV_ID FROM ' || my_table || 
	        ' WHERE MASK = 0';
             EXECUTE IMMEDIATE sql_str INTO zero_iov;
             dbms_output.put_line('Found IOV_ID = ' || zero_iov || 
                ' with mask = 0');
             -- still we have to update tables
	     j := 1;
             WHILE j <= my_mask LOOP
                -- we loop on tables looking for the just changed IOV_ID and modify it as the last one
	        sql_str := 'SELECT COUNT(TABLE_NAME) FROM ' || my_dattable || 
                   ' WHERE BITAND(BITAND(ID, :j), :my_mask) > 0';
                EXECUTE IMMEDIATE sql_str INTO tnc USING j, my_mask;
                IF tnc > 0 THEN
  	           sql_str := 'SELECT TABLE_NAME FROM ' || 
                      my_dattable || 
                      ' WHERE BITAND(BITAND(ID, :j), :my_mask) > 0';
                   EXECUTE IMMEDIATE sql_str INTO tn USING j, my_mask;
                   dbms_output.put_line('Found table ' || tn || 
                      ' to be updated');
                   dbms_output.put_line('      Setting IOV_ID = ' || 
                      zero_iov || ' from ' || last_iov);
	           sql_str := 'UPDATE ' || tn || ' SET IOV_ID = :zero_iov WHERE IOV_ID = '
                      || ':last_iov';
                   EXECUTE IMMEDIATE sql_str USING zero_iov, last_iov;
                   dbms_output.put_line(sql_str);
                END IF;
                j := j * 2;
	     END LOOP;
             -- remask last insert id
             sql_str := 'UPDATE ' || my_table || ' SET MASK = :my_mask WHERE MASK = 0';
             EXECUTE IMMEDIATE sql_str USING my_mask;	
	     dbms_output.put_line(sql_str || ' using ' || my_mask);
          ELSE
           -- update the till of this last measurement
             sql_str := 'UPDATE ' || my_table || ' SET TILL = :new_start ' || 
	        ' WHERE IOV_ID = :last_iov';
             EXECUTE IMMEDIATE sql_str USING new_start, last_iov;
	     dbms_output.put_line('Updated IOV ' || last_iov || ' Set TILL = ' ||
		new_start);
          END IF;
       END LOOP;
    END IF;
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
