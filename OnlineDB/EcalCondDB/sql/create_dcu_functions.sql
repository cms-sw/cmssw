/* the following commands provide a function that returns
   a table containing the required DCU data considered valid
   at a given time, without taking into account the EOV,
   since it is not properly set, by definition. */  

/* create the types needed to define the result of the
   function */
CREATE OR REPLACE TYPE T_DCU_SELECT AS OBJECT (
	LOGIC_ID INTEGER,
	IOV_ID   INTEGER,
	TIME     DATE,
	VALUE    FLOAT
);
/

CREATE OR REPLACE TYPE T_TABLE_DCU_SELECT AS TABLE OF T_DCU_SELECT;
/

/* create the function 
   usage: SELECT * FROM TABLE(DCU_SELECT(table, column, date)
   The date must be a string formatted as 'DD-MM-YYYY HH24:MI:SS' */
CREATE OR REPLACE FUNCTION DCU_SELECT
( 
  tblname IN VARCHAR2,
  column IN VARCHAR2,
  time IN VARCHAR2)  RETURN T_TABLE_DCU_SELECT PIPELINED IS

  TYPE      ref0 IS REF CURSOR;

  sql_str   VARCHAR(1000);
  logic_id  INTEGER;
  channels  INTEGER;
  value     FLOAT;
  since     DATE;
  cur0      ref0;	

  V_RET     T_DCU_SELECT
         := T_DCU_SELECT(NULL, NULL, NULL, NULL);

  BEGIN
    -- evaluate how many channels there are in the required table 
    sql_str := 'SELECT COUNT(LOGIC_ID) FROM (SELECT DISTINCT LOGIC_ID FROM ' ||
	tblname || ')';
    EXECUTE IMMEDIATE sql_str INTO channels;
    -- get the value of the required field whose start of validity is lower
    -- than the required date (limit the number of days considered)
    OPEN cur0 FOR 
        'SELECT LOGIC_ID, IOV_ID, TIME, VALUE FROM (SELECT LOGIC_ID, ' ||
	'MAX(D.IOV_ID) IOV_ID, MAX(SINCE) TIME, '
        || column || ' VALUE FROM ' || tblname || 
	' D JOIN DCU_IOV R ON D.IOV_ID = R.IOV_ID WHERE' ||
	' SINCE <= TO_DATE(:1, ''DD-MM-YYYY HH24:MI:SS'') ' ||
	' AND SINCE >= (TO_DATE(:2, ''DD-MM-YYYY HH24:MI:SS'') - 7)' || 
	' GROUP BY LOGIC_ID, ' || column || 
	' ORDER BY TIME DESC) WHERE ROWNUM <= :3'
    USING time, time, channels;
    LOOP
      FETCH cur0 INTO V_RET.LOGIC_ID, V_RET.IOV_ID, V_RET.TIME, V_RET.VALUE;
      EXIT WHEN cur0%NOTFOUND;
      PIPE ROW(V_RET);
    END LOOP;
    CLOSE cur0; 
    RETURN;	
  END DCU_SELECT;
/
