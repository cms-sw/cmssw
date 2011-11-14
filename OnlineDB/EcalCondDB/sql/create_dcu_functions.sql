/* the following commands provide a function that returns
   a table containing the required DCU data considered valid
   at a given time, without taking into account the EOV,
   since it is not properly set, by definition. */  

/* create the types needed to define the result of the
   function */
CREATE OR REPLACE TYPE T_DCU_SELECT AS OBJECT (
	LOGIC_ID INTEGER,
	SINCE    DATE,
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
  time IN VARCHAR2)  RETURN T_TABLE_DCU_SELECT IS

  sql_str   VARCHAR(1000);
  logic_id  INTEGER;
  channels  INTEGER;
  value     FLOAT;
  since     DATE;
  i         INTEGER;

  V_RET     T_TABLE_DCU_SELECT;

  BEGIN
    -- create the table
    V_RET := T_TABLE_DCU_SELECT();
    
    -- evaluate how many channels there are in the required table 
    sql_str := 'SELECT COUNT(*) FROM (SELECT DISTINCT LOGIC_ID FROM ' ||
	tblname || ')';
    EXECUTE IMMEDIATE sql_str INTO channels;
    dbms_output.enable(10000000);	
    FOR i IN 1..channels LOOP
     BEGIN
       -- for each channel get the logic_id	
       sql_str := 'SELECT LOGIC_ID FROM (SELECT DISTINCT ROWNUM I, ' || 
	'LOGIC_ID FROM ' || tblname || ') WHERE I = :i';
       EXECUTE IMMEDIATE sql_str INTO logic_id USING i;
       -- get the value of the required field whose start of validity is lower
       -- than the required date
       sql_str := 'SELECT * FROM (SELECT LOGIC_ID, SINCE, ' || column || 
	' VALUE FROM ' || tblname || 
	' D JOIN DCU_IOV R ON D.IOV_ID = R.IOV_ID WHERE LOGIC_ID = :1 AND ' ||
	' SINCE <= TO_DATE(''' || time || ''', ''DD-MM-YYYY HH24:MI:SS'') ' || 
	' ORDER BY SINCE DESC) WHERE ROWNUM <= 1';
       EXECUTE IMMEDIATE sql_str INTO logic_id, since, value USING logic_id;	
       -- extend and fill the resulting table
       V_RET.EXTEND;	
       V_RET(V_RET.COUNT) := T_DCU_SELECT(logic_id, since, value);
       EXCEPTION
	 WHEN NO_DATA_FOUND THEN
            -- DO NOTHING
	    NULL;
     END;	
    END LOOP;
    return V_RET;	
  END DCU_SELECT;
/
