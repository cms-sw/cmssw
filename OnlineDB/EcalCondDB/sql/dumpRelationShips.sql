/*

  This script generates two files with a complete description of tables
  and their foreign key relationships

  Author: Giovanni.Organtini@roma1.infn.it 2011

*/

SET NEWPAGE 0
SET SPACE 0
SET LINESIZE 250
SET PAGESIZE 0
SET ECHO OFF
SET FEEDBACK OFF
SET HEADING OFF
SET MARKUP HTML OFF
SET LONG 90000;

column parent format a30
column child format a30
column column_name format a30

/* tables */
SPOOL TABLES.DAT
SELECT T.TABLE_NAME, COLUMN_NAME FROM USER_TABLES T JOIN USER_TAB_COLUMNS C ON
T.TABLE_NAME = C.TABLE_NAME ORDER BY TABLE_NAME;
SPOOL OFF

/* constraints */
SPOOL RELATIONSHIPS.DAT
SELECT a.table_name parent, a.column_name, c.table_name child, c.column_name
FROM user_cons_columns a,user_cons_columns c,user_constraints b
WHERE a.constraint_name=b.constraint_name
AND a.table_name=b.table_name
AND b.constraint_type='R'
AND b.r_constraint_name=c.constraint_name
ORDER by a.table_name;
SPOOL OFF

