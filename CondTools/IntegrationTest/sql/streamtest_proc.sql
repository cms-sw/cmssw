DROP TYPE n_tbl;
DROP TYPE n_obj;


CREATE OR REPLACE TYPE n_obj AS OBJECT (
 n NUMBER
);
/
show errors;

CREATE OR REPLACE TYPE n_tbl AS TABLE OF n_obj;
/
show errors;

CREATE OR REPLACE FUNCTION
nrows (n IN NUMBER DEFAULT NULL, m IN NUMBER DEFAULT NULL)
return n_tbl
PIPELINED
AS
nrow n_obj := n_obj(0);
BEGIN
  FOR i IN nvl(n,1) .. nvl(m,999999999)
  LOOP
    nrow.n := i;
    PIPE ROW(nrow);
  END LOOP;
  RETURN;
END;
/
show errors;

SELECT n FROM TABLE(nrows(1,5));

CREATE OR REPLACE PROCEDURE streamtest (n_objects IN NUMBER)
AS
last_id NUMBER(10);
n_start NUMBER(10);
n_end NUMBER(10);
BEGIN
  dbms_output.put_line('Streamtest adding ' || n_objects || ' objects');
  SELECT max(iov_value_id) INTO last_id FROM st_ecalpedestals;

  IF last_id IS NULL THEN
    last_id := 0;
  END IF;

  n_start := last_id + 1;
  n_end := n_start + n_objects - 1;

  dbms_output.put_line('Writing objects ' || n_start || ' through ' || n_end);

  INSERT INTO st_ecalpedestals (iov_value_id, time)
  SELECT n, n FROM TABLE(nrows(n_start, n_end));

  INSERT INTO st_ecalpedestals_item 
  (iov_value_id, pos, det_id, mean_x1, mean_x12, mean_x6, rms_x1, rms_x12, rms_x6)
  SELECT par_key.n, pos_key.n, dbms_random.random(), 
         dbms_random.value(), dbms_random.value(), dbms_random.value(),
         dbms_random.value(), dbms_random.value(), dbms_random.value()
  FROM TABLE(nrows(n_start, n_end)) par_key, TABLE(nrows(1, 61200)) pos_key;

  COMMIT;
END;
/
show errors;
