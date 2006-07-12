/* Utterly annihilate a DB */

BEGIN
  FOR result IN (SELECT table_name FROM user_tables)
  LOOP
    EXECUTE IMMEDIATE 'DROP TABLE ' || result.table_name || ' CASCADE CONSTRAINTS';
  END LOOP;
END;
/

BEGIN
  FOR result IN (SELECT sequence_name FROM user_sequences)
  LOOP
    EXECUTE IMMEDIATE 'DROP SEQUENCE ' || result.sequence_name;
  END LOOP;
END;
/

BEGIN
  FOR result IN (SELECT object_name FROM user_procedures)
  LOOP
    EXECUTE IMMEDIATE 'DROP PROCEDURE ' || result.object_name;
  END LOOP;
END;
/


BEGIN
  FOR result IN (SELECT object_name AS FUNCTION FROM user_objects WHERE object_type = 'FUNCTION')
  LOOP
    EXECUTE IMMEDIATE 'DROP FUNCTION ' || result.function;
  END LOOP;
END;
/


BEGIN
  FOR result IN (SELECT trigger_name, table_name FROM user_triggers WHERE trigger_name NOT LIKE 'BIN%')
  LOOP
    EXECUTE IMMEDIATE 'DROP TRIGGER ' || result.trigger_name;
  END LOOP;
END;
/


BEGIN
  FOR result IN (SELECT db_link, username FROM user_db_links)
  LOOP
    EXECUTE IMMEDIATE 'DROP DATABASE LINK ' || result.db_link;
  END LOOP;
END;
/
