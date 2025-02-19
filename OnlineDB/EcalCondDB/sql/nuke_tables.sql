BEGIN
  FOR result IN (SELECT table_name FROM user_tables)
  LOOP
    EXECUTE IMMEDIATE 'DROP TABLE ' || result.table_name || ' CASCADE CONSTRAINTS';
  END LOOP;
END;
/
