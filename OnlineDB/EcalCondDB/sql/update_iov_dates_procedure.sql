/* $id$ 
 * 
 * Procedure to validate an IoV to be inserted and update a previous
 * IoV so that there are no overlaps.
 */
CREATE OR REPLACE PROCEDURE update_iov_dates
( my_table IN VARCHAR2,
  start_col IN VARCHAR2,
  end_col IN VARCHAR2,
  new_start IN DATE,
  new_end IN OUT DATE,
  new_tag_id IN NUMBER ) IS
    
  sql_str VARCHAR(1000);
  future_start DATE;
  last_start DATE;
  last_end DATE;

  BEGIN
    -- Ensure IoV time has positive duration
    IF new_end <= new_start THEN
       raise_application_error(-20000, 'IOV must have ' || start_col || ' < ' || end_col);
    END IF;

    -- Truncate for IoVs in the future of this one
    -- Fetch IOV immediately after this one 
    sql_str := 'SELECT min(' || start_col || ') FROM ' || my_table || 
               ' WHERE ' || start_col || ' > :s AND tag_id = :t';
    EXECUTE IMMEDIATE sql_str INTO future_start USING new_start, new_tag_id;

    IF future_start IS NOT NULL THEN
      -- truncate this IOV
      new_end := future_start;
    END IF;

    -- Fetch the most recent IoV prior to this one
    sql_str := 'SELECT max(' || start_col || ') FROM ' || my_table || 
               ' WHERE ' || start_col || ' <= :s AND tag_id = :t';
    EXECUTE IMMEDIATE sql_str INTO last_start USING new_start, new_tag_id;

    IF last_start IS NULL THEN
        -- table has no previous data for this tag, nothing to do
        return;
    END IF;

    -- Fetch the end of this most recent IoV
    sql_str := 'SELECT ' || end_col || ' FROM ' || my_table || ' WHERE ' || start_col || ' = :s AND tag_id = :t';
    EXECUTE IMMEDIATE sql_str INTO last_end USING last_start, new_tag_id;

    IF new_start = last_start THEN
        -- Attempted to insert overlapping IoV!
        raise_application_error(-20020, 'New IOV ''' || start_col || ''' overlaps older ''' || start_col || '''');
    ELSIF new_start < last_end THEN
       -- Truncate the last IoV
       sql_str := 'UPDATE ' || my_table || ' SET ' || end_col || ' = :new_start' || 
                  ' WHERE ' || start_col || ' = :last_start AND ' || end_col || ' = :last_end AND tag_id = :t';
       EXECUTE IMMEDIATE sql_str USING new_start, last_start, last_end, new_tag_id;
    END IF;
  END;
/
show errors;
