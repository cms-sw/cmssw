/* the following commands provide a function that returns
   a table containing column names and corresponding types.
   It is needed since the USER_TAB_COLS is not visible to
   other accounts. */	

/* create the types needed to define the result of the
   function */
CREATE OR REPLACE TYPE T_LMF_COLS AS OBJECT (
	COLUMN_NAME VARCHAR(25),
	DATA_TYPE   VARCHAR(25)
);
/

CREATE OR REPLACE TYPE T_TABLE_LMF_COLS AS TABLE OF T_LMF_COLS;
/

/* create the function 
   usage: SELECT * FROM TABLE(LMF_TAB_COLS(table, column_to_exclude))
*/
CREATE OR REPLACE FUNCTION LMF_TAB_COLS
( 
  tblname IN VARCHAR2,
  iovFieldName IN VARCHAR2) RETURN T_TABLE_LMF_COLS PIPELINED IS

  TYPE      ref0 IS REF CURSOR;

  sql_str   VARCHAR(1000);
  cur0      ref0;	

  V_RET     T_LMF_COLS
         := T_LMF_COLS(NULL, NULL);

  BEGIN
    OPEN cur0 FOR 
        'SELECT COLUMN_NAME, DATA_TYPE FROM USER_TAB_COLS WHERE ' ||
	'TABLE_NAME = :1 AND COLUMN_NAME != ''LOGIC_ID'' AND COLUMN_NAME != :2'
    USING tblname, iovFieldName;
    LOOP
      FETCH cur0 INTO V_RET.COLUMN_NAME, V_RET.DATA_TYPE;
      EXIT WHEN cur0%NOTFOUND;
      PIPE ROW(V_RET);
    END LOOP;
    CLOSE cur0; 
    RETURN;	
  END LMF_TAB_COLS;
/

GRANT EXECUTE ON LMF_TAB_COLS TO CMS_ECAL_R
/
