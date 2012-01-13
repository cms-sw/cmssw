/******************************************************************************

This script is used to create the scripts that actually splits partitions
on CMS_ECAL_COND account

******************************************************************************/

column pind format a2
column table_name format a30
column column_name format a30
column pname format a30

DROP TABLE SPLITTER;

/* 
   Create a table which lists all the partitioned tables.
   For each partitioned table stores the name, the column on which
   it has been partitioned, the partition name and the index of the 
   partition. Our partition names are of the form XXXX_n, where n
   is a progressive number. This number is the index of the
   partition. The current partition is the one with the highest
   index
*/
CREATE TABLE SPLITTER AS 
select table_name, column_name, pname, max(pind) pind, num_rows 
	from (select t.table_name,
	column_name,
	regexp_replace(partition_name, '_[0-9]+$', '') pname, 
	regexp_replace(partition_name, '.*_', '') pind, u.num_rows num_rows
	from 
	user_tab_partitions t join user_part_key_columns c on t.table_name =
	c.name join user_tables u on t.table_name = u.table_name) 
        group by pname, table_name, column_name, num_rows order by num_rows asc;

/*
   Add a column to that table to store the maximum value assigned
   to the partitioning column.
*/
ALTER TABLE SPLITTER ADD MAXVAL INT DEFAULT 0;

COLUMN A FORMAT A30 

SET HEAD OFF
SET ECHO OFF
/*
   Create a script to generate SQL commands to generate another
   script to update the MAXVAL column of the SPLITTER table. Use 
   the COALESCE function to fight against empty tables.
*/
column a format a35
SPOOL UPDATESPLITTER1.sql
SELECT 'SELECT ''UPDATE SPLITTER SET MAXVAL = '', COALESCE(MAX(', 
	COLUMN_NAME, '),0), '' WHERE TABLE_NAME = ', 
	REGEXP_REPLACE(REGEXP_REPLACE(TABLE_NAME, '^ *', ''''''), 
	' *$', '''''') A,
	';'' FROM ', TABLE_NAME, ';'
	FROM SPLITTER;
SPOOL OFF

/*
   Run the generated script, to generate the script to actually 
   update SPLITTER.
*/
SPOOL UPDATESPLITTER2.sql;
@UPDATESPLITTER1;
SPOOL OFF

/* 
   update SPLITTER with the current values for partitioning columns
*/
@UPDATESPLITTER2;

/*
   generate the sql instructions to partition tables
*/
SET LINE 400
SPOOL SPLITPARTITIONS.sql
SELECT 'ALTER TABLE ' || TABLE_NAME || ' SPLIT PARTITION ' || PNAME || '_' ||
	PIND || ' AT (' || MAXVAL || ') INTO (PARTITION ' || PNAME || '_' || 
	(PIND) || ' TABLESPACE CMS_ECAL_COND_20' || PIND || '_DATA' ||
	', PARTITION ' || PNAME || '_' || (PIND + 1) || 
	' TABLESPACE CMS_ECAL_COND_20' || (PIND+1) || 
	'_DATA) UPDATE GLOBAL INDEXES;' 
	FROM SPLITTER WHERE MAXVAL > 0 ORDER BY NUM_ROWS ASC;
SPOOL OFF

/* do the splitting (to be done by hand)
@SPLITPARTITIONS;
*/

/*
SELECT table_name, partition_name, high_value
FROM user_tab_partitions;
*/