q/* initial setup */
SET NEWPAGE 0
SET SPACE 0
SET LINESIZE 250
SET PAGESIZE 0
SET ECHO OFF
SET FEEDBACK OFF
SET HEADING OFF
SET MARKUP HTML OFF
SET LONG 90000;

/* create instructions to recreate the tables */
spool ddl_list.sql
  select 'select dbms_metadata.get_ddl(''TABLE'', ', 
	  concat(concat('''', regexp_replace(table_name, ' *', '')),''''), 
          ', ''CMS_ECAL_COND'') from dual;' 
   from 	user_tables;
  select 'spool pks.data' from dual;
  select 'select t.table_name, column_name from user_tab_columns t join ',
	 'user_constraints c on t.table_name = c.table_name where ',
	 'column_name like ''%IOV_ID%'' and constraint_type = ''P'';' from dual;
  select 'spool off' from dual;		
spool off;

/* list triggers: not used but we profit to save them */
SPOOL triggers.data;
SELECT TRIGGER_NAME, TRIGGER_TYPE, TABLE_NAME, REFERENCING_NAMES, ACTION_TYPE, \
TRIGGER_BODY FROM USER_TRIGGERS;

SPOOL OFF;
/

/* list functions: not used but we profit to save them */
SPOOL functions.data;
SELECT NAME, LINE, TEXT FROM USER_SOURCE WHERE TYPE = 'FUNCTION';

SPOOL OFF;
/

/* list procedures: not used but we profit to save them */
SPOOL procedures.data;
SELECT NAME, LINE, TEXT FROM USER_SOURCE WHERE TYPE = 'PROCEDURE';

SPOOL OFF;
/
/* list indexes */
SPOOL indexes.data;
COLUMN COLUMN_NAME FORMAT A35;
COLUMN TABLE_NAME FORMAT A35;
COLUMN INDEX_NAME FORMAT A35;
SELECT I.INDEX_NAME, I.TABLE_NAME, C.COLUMN_NAME FROM USER_INDEXES I JOIN
        USER_IND_COLUMNS C ON I.INDEX_NAME = C.INDEX_NAME;
SPOOL OFF;
/

/*                                                                    
 get the current size of each user table as well as                  
 the name of the IOV type column                                          
*/

CREATE TABLE TSIZE AS
SELECT T.TABLE_NAME TNAME, T.COLUMN_NAME COLNAME,
        (S.BYTES/1024/1024/1024) GB
             FROM USER_TAB_COLS T, USER_ALL_TABLES A, USER_SEGMENTS S
             WHERE T.TABLE_NAME = A.TABLE_NAME
                AND
               (T.COLUMN_NAME LIKE '%IOV%' AND T.DATA_TYPE LIKE '%NUMBER%')
               AND S.SEGMENT_NAME = T.TABLE_NAME ORDER BY BYTES ASC
/

SELECT * FROM TSIZE;

/* create a script that gets info from the tables */
SPOOL GETIOVS.sql
SELECT 'SELECT ''', TNAME, ' ', COLNAME, ' ', GB, ''', MAX(', COLNAME, ') S ',
        'FROM ',
        TNAME , ';'
        FROM TSIZE;
SPOOL OFF

DROP TABLE TSIZE;

/* write results to a text file */
SPOOL partition.data
@GETIOVS
SPOOL OFF

/* create script to recreate the tables */
spool recreate.sql
@ddl_list
spool off
/

/* disable constraints */
SPOOL DISABLECONSTRAINTS.sql

SELECT 'SET ECHO ON;' FROM DUAL;
SELECT 'ALTER TABLE ', TABLE_NAME, ' DISABLE CONSTRAINT ', CONSTRAINT_NAME,
        ' CASCADE;' FROM
        USER_CONSTRAINTS WHERE TABLE_NAME NOT LIKE 'BIN%' AND CONSTRAINT_NAME
	LIKE '%PK';

SELECT 'ALTER TABLE ', TABLE_NAME, ' DISABLE CONSTRAINT ', CONSTRAINT_NAME,
	' CASCADE;' FROM
        USER_CONSTRAINTS WHERE TABLE_NAME NOT LIKE 'BIN%' AND CONSTRAINT_NAME
        LIKE '%FK';
SELECT 'SET ECHO OFF;' FROM DUAL;

SPOOL OFF;
/

/* re-enable constraints */
SPOOL ENABLECONSTRAINTS.sql

SELECT 'SET ECHO ON;' FROM DUAL;
SELECT 'ALTER TABLE ', TABLE_NAME, ' ENABLE CONSTRAINT ', CONSTRAINT_NAME,
        ';' FROM
        USER_CONSTRAINTS WHERE TABLE_NAME NOT LIKE 'BIN%' AND CONSTRAINT_NAME
	LIKE '%PK';

SELECT 'ALTER TABLE ', TABLE_NAME, ' ENABLE CONSTRAINT ', CONSTRAINT_NAME,
	';' FROM
        USER_CONSTRAINTS WHERE TABLE_NAME NOT LIKE 'BIN%' AND CONSTRAINT_NAME
        LIKE '%FK';
SELECT 'SET ECHO OFF;' FROM DUAL;

SPOOL OFF;
/

