/* Show all the database objects */

SELECT table_name FROM user_tables;
SELECT sequence_name FROM user_sequences;
SELECT procedure_name, object_name FROM user_procedures;
SELECT trigger_name, table_name FROM user_triggers WHERE trigger_name NOT LIKE 'BIN%';
