/* Stores the schema name and top-level-table of an object that undergoes O2O */

CREATE TABLE o2o_setup (
  object_name VARCHAR2(32),
  schema VARCHAR(32),
  top_level_table VARCHAR2(32),
  PRIMARY KEY (object_name)
);

