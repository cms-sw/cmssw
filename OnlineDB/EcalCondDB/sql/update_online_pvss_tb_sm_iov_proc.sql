CREATE OR REPLACE
PROCEDURE update_online_pvss_tb_sm_iov
( cndc_table IN VARCHAR2,
  new_since IN DATE,
  new_till IN DATE,
  dp_name IN VARCHAR2 ) IS
/* $id$
 *
 * Procedure to validate an IoV to be inserted and update a previous
 * IoV so that there are no overlaps.
 */
  sql_str VARCHAR(1000);
  future_since DATE;
  last_since DATE;
  last_till DATE;

  BEGIN
    -- Ensure IoV has positive duration
    IF new_till <= new_since THEN
       raise_application_error(-20000, 'IoV must have since < till');
    END IF;

    -- Make sure that there are no IoVs in the future of this one
    sql_str := 'SELECT max(since) FROM ' || cndc_table || ' WHERE since > :s and dp_name = :d';
    EXECUTE IMMEDIATE sql_str INTO future_since USING new_since, dp_name;

    IF future_since IS NOT NULL THEN
      raise_application_error(-20010, 'IoVs must be inserted in increasing order:  ' ||
                                      'Current highest ''since'' is ' ||
                                      to_char(future_since, 'YYYY-MM-DD HH24:MI:SS'));
    END IF;

    -- Fetch the most recent IoV prior to this one
    sql_str := 'SELECT max(since) FROM ' || cndc_table || ' WHERE since <= :s and dp_name = :d';
    EXECUTE IMMEDIATE sql_str INTO last_since USING new_since, dp_name;

    IF last_since IS NULL THEN
        -- table has no data, nothing to do
        return;
    END IF;

    -- Fetch the till of this most recent IoV
    sql_str := 'SELECT till FROM ' || cndc_table || ' WHERE since = :s and dp_name = :d';
    EXECUTE IMMEDIATE sql_str INTO last_till USING last_since, dp_name;

    IF new_since = last_since THEN
        -- Attempted to insert overlapping IoV!
        raise_application_error(-20020, 'New IOV since overlaps older since');
    ELSIF new_since < last_till THEN
       -- Truncate the last IoV
       sql_str := 'UPDATE ' || cndc_table || ' SET till = :new_since WHERE since = :last_since AND till = :last_till and dp_name = :d';
       EXECUTE IMMEDIATE sql_str USING new_since, last_since, last_till, dp_name;
    END IF;
  END;
/
show errors;

CREATE OR REPLACE
TRIGGER pvss_tb_sm_iov_tg
  BEFORE INSERT ON pvss_tb_sm_dat
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_tb_sm_iov('pvss_tb_sm_dat', :newiov.since, :newiov.till,:newiov.dp_name);
end;
/
