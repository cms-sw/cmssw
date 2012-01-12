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
SET LINES 230
START title132 'Table Partition File Storage'
SELECT
     table_name,
     tablespace_name,
     partition_name,
     partition_position,
     high_value,
     pct_free,
     pct_used,
     ini_trans,
     max_trans,
     initial_extent,
     next_extent,
     max_extent,
     pct_increase
FROM USER_tab_partitions
ORDER BY table_name, partition_position
/
