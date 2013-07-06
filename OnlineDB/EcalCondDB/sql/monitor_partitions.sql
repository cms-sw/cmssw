COLUMN table_name         FORMAT a30        HEADING 'Table'
COLUMN partition_name     FORMAT a30         HEADING 'Partition'
COLUMN tablespace_name    FORMAT a30        HEADING 'Tablespace'
COLUMN pct_free           FORMAT 9999       HEADING '%|Free'
COLUMN pct_used           FORMAT 999        HEADING '%|Use'
COLUMN ini_trans          FORMAT 9999       HEADING 'Init|Tran'
COLUMN max_trans          FORMAT 9999       HEADING 'Max|Tran'
COLUMN initial_extent     FORMAT 9999999    HEADING 'Init|Extent'
COLUMN next_extent        FORMAT 9999999    HEADING 'Next|Extent'
COLUMN max_extent                           HEADING 'Max|Extents'
COLUMN pct_increase       FORMAT 999        HEADING '%|Inc'
COLUMN partition_position FORMAT 9999       HEADING 'Part|Nmbr'
COLUMN high_value         FORMAT a8         HEADING 'High|Value'
COLUMN column_name        FORMAT a15        HEADING 'key'

CREATE TABLE TEMP_TABLE (
	TABLE_NAME VARCHAR2(35),
	MAX INT
)
/

set term off
SPOOL TEMP.sql
set head off
set echo off
set feed off
SELECT DISTINCT 'INSERT INTO TEMP_TABLE (TABLE_NAME, MAX) SELECT ''' || 
	TABLE_NAME || ''' TABLE_NAME, COALESCE(MAX(' || 
	COLUMN_NAME || '),0) MAX FROM ' || 
	TABLE_NAME || ';' STATEMENT FROM USER_TAB_PARTITIONS P JOIN
        USER_PART_KEY_COLUMNS C ON P.TABLE_NAME = C.NAME; 
SPOOL OFF
@TEMP

SET LINES 230
SET TERM ON
SET HEAD ON
SET FEED ON
SELECT 
     p.table_name,
     tablespace_name,
     partition_name,
     partition_position,
     column_name, 
     high_value,
     t.max, 
     pct_free,
     pct_used,
     ini_trans,
     max_trans,
     initial_extent,
     next_extent,
     max_extent,
     pct_increase
FROM USER_tab_partitions p JOIN USER_part_key_columns c ON 
     p.TABLE_NAME = c.NAME JOIN TEMP_TABLE t ON p.TABLE_NAME =
     t.TABLE_NAME
/*WHERE partition_name like '%_0' and partition_name not like '%_10'*/
ORDER BY table_name, partition_position
/

drop table TEMP_TABLE
/


