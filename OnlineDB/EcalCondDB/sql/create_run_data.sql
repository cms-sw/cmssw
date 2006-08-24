/*
 *  Creates all the data tables referencing RUN_IOV
 */


CREATE TABLE run_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  num_events		NUMBER(10)
);

ALTER TABLE run_dat ADD CONSTRAINT run_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_dat ADD CONSTRAINT run_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);



CREATE TABLE run_config_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  config_tag		VARCHAR2(100) NOT NULL,
  config_ver		NUMBER(10) NOT NULL
);

ALTER TABLE run_config_dat ADD CONSTRAINT run_config_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_config_dat ADD CONSTRAINT run_config_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);



CREATE TABLE run_h4_table_position_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  table_x		NUMBER(10),
  table_y		NUMBER(10),
  number_of_spills	NUMBER(10),
  number_of_events	NUMBER(10)
);

ALTER TABLE run_h4_table_position_dat ADD CONSTRAINT run_h4_table_position_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_h4_table_position_dat ADD CONSTRAINT run_h4_table_position_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);
