/*
 * This script will partition the whole database, based on IOV_IDs
 *
 * It creates a script that contains the actual commands to partition
 * all the tables that contain some key to an IOV_ID.
 *
 * Partitions are created based on a division of IOV_IDs in blocks
 * of N, up to a maximum value M. Adjust these values using variables
 * step and imax in the DECLARE block below. 
 *
 * Usage:
 * 1. Run this script with the command @partition_db
 * 2. Check that a new MKPARTITION.sql script and a fresh SPLIT.sql script
 *    are created and they contain reasonable values
 * 3. Run @MKPARTITION, crossing your fingers and spelling verses
 *    from any ORACLE tutorial
 * 
 * Copyright (C) by Giovanni.Organtini@roma1.infn.it 2009
 *
 */

/* basic steup */
SET NEWPAGE 0
SET SPACE 0
SET LINESIZE 80
SET PAGESIZE 0
SET ECHO OFF
SET FEEDBACK OFF
SET HEADING OFF
SET MARKUP HTML OFF

/* generate a script that in turn generate the commands to split the tables */
SPOOL SPLIT.sql

CREATE TABLE DUMMY (
	MYQUERY VARCHAR(1000)
)
/

BEGIN
DECLARE 
    step NUMBER :=100;
    imax NUMBER :=1000;
    i NUMBER :=step;
    q VARCHAR(1000) := '';
 BEGIN
  LOOP
  q := 'SELECT "ALTER TABLE ", REGEXP_REPLACE(TNAME, "$", ""), 
	  " SPLIT PARTITION ", REGEXP_REPLACE(TNAME, "$", "_0"), 
	  " AT (", i, ") INTO (PARTITION ", REGEXP_REPLACE(TNAME, "$", "_1"),
	  ", PARTITION ", REGEXP_REPLACE(TNAME, "$", "_0"), 
	  ") UPDATE GLOBAL INDEXES;" FROM 
	    (SELECT T.TABLE_NAME TNAME, T.COLUMN_NAME COLNAME 
             FROM USER_TAB_COLS T, USER_ALL_TABLES A
	     WHERE T.TABLE_NAME = A.TABLE_NAME AND 
               (T.COLUMN_NAME LIKE "%IOV%" AND T.DATA_TYPE LIKE "%NUMBER%")
            ) WHERE TNAME NOT LIKE "BIN%";';
	q := REGEXP_REPLACE(q, 'i', i); 
	q := REGEXP_REPLACE(q, '"', ''''); 
	insert into dummy values (q);
   i := i+step;
   EXIT WHEN i > imax;
   END LOOP;
 END;
END;
/

SELECT * FROM DUMMY;

DROP TABLE DUMMY;

SPOOL OFF

/* generate a script that in fact create partitions */
SPOOL MKPARTITIONS.sql

/* first of all create new tables from existing ones, with a single partition */
SELECT 'CREATE TABLE ',REGEXP_REPLACE(TNAME, '$', '_2'), 
	' AS SELECT * FROM ', REGEXP_REPLACE(TNAME, '$', ''),
	' PARTITION BY RANGE (', REGEXP_REPLACE(COLNAME, ' +$', ''), 
	') (PARTITION ', REGEXP_REPLACE(TNAME, '$', '_0'), 
	' VALUES LESS THAN (MAXVALUE));'
            FROM 
	    (SELECT T.TABLE_NAME TNAME, T.COLUMN_NAME COLNAME 
             FROM USER_TAB_COLS T, USER_ALL_TABLES A
	     WHERE T.TABLE_NAME = A.TABLE_NAME AND 
               (T.COLUMN_NAME LIKE '%IOV%' AND T.DATA_TYPE LIKE '%NUMBER%')
            ) WHERE TNAME NOT LIKE 'BIN%';

/* then drop the existing tables and rename the new ones */

SELECT 'DROP TABLE ', REGEXP_REPLACE(TNAME, '$', '') NEWTNAME, 
	'; RENAME', REGEXP_REPLACE(TNAME, '$', '_2'), ' TO ',
        REGEXP_REPLACE(TNAME, '$', '')	FROM 
	    (SELECT T.TABLE_NAME TNAME, T.COLUMN_NAME COLNAME 
             FROM USER_TAB_COLS T, USER_ALL_TABLES A
	     WHERE T.TABLE_NAME = A.TABLE_NAME AND 
               (T.COLUMN_NAME LIKE '%IOV%' AND T.DATA_TYPE LIKE '%NUMBER%')
            ) WHERE TNAME NOT LIKE 'BIN%';

/* finally call the script that will create partitions, in fact */

@SPLIT

SPOOL OFF


