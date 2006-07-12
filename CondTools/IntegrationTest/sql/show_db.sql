/* Show all the database objects */

SELECT table_name FROM user_tables ORDER BY table_name;
SELECT sequence_name FROM user_sequences ORDER BY sequence_name;
SELECT procedure_name, object_name FROM user_procedures ORDER BY procedure_name;
SELECT object_name AS FUNCTION FROM user_objects WHERE object_type = 'FUNCTION';
SELECT trigger_name, table_name FROM user_triggers WHERE trigger_name NOT LIKE 'BIN%' ORDER BY trigger_name;
SELECT db_link, username FROM user_db_links;
