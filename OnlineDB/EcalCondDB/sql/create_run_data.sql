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

CREATE TABLE run_tpgconfig_dat (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10),
  config_tag            VARCHAR2(100) NOT NULL,
  version NUMBER(10) NOT NULL 
);

 ALTER TABLE run_TPGCONFIG_DAT ADD CONSTRAINT run_tpgconfig_dat_pk PRIMARY KEY (iov_id, logic_id);
 ALTER TABLE run_TPGCONFIG_DAT ADD CONSTRAINT run_tpgconfig_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);



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




-- Tables for channel masking
CREATE TABLE run_crystal_errors_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  error_bits		NUMBER(38)
);

ALTER TABLE run_crystal_errors_dat ADD CONSTRAINT run_crystal_errors_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_crystal_errors_dat ADD CONSTRAINT run_crystal_errors_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);




CREATE TABLE run_tt_errors_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  error_bits		NUMBER(38)
);

ALTER TABLE run_tt_errors_dat ADD CONSTRAINT run_tt_errors_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_tt_errors_dat ADD CONSTRAINT run_tt_errors_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);



CREATE TABLE run_pn_errors_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  error_bits		NUMBER(38)
);

ALTER TABLE run_pn_errors_dat ADD CONSTRAINT run_pn_errors_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_pn_errors_dat ADD CONSTRAINT run_pn_errors_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);



CREATE TABLE run_mem_ch_errors_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  error_bits		NUMBER(38)
);

ALTER TABLE run_mem_ch_errors_dat ADD CONSTRAINT run_mem_ch_errors_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_mem_ch_errors_dat ADD CONSTRAINT run_mem_ch_errors_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);



CREATE TABLE run_mem_tt_errors_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  error_bits		NUMBER(38)
);

ALTER TABLE run_mem_tt_errors_dat ADD CONSTRAINT run_mem_tt_errors_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_mem_tt_errors_dat ADD CONSTRAINT run_mem_tt_errors_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);



CREATE TABLE error_dictionary_def (
  def_id		NUMBER(10),
  error_mask		NUMBER(38),
  short_desc		VARCHAR2(100),
  long_desc		VARCHAR2(1000)
);

CREATE SEQUENCE error_dictionary_def_sq INCREMENT BY 1 START WITH 1;
ALTER TABLE error_dictionary_def ADD CONSTRAINT error_dictionary_def_pk PRIMARY KEY (def_id);
ALTER TABLE error_dictionary_def ADD CONSTRAINT error_dictionary_def_uk UNIQUE (error_mask);
CREATE INDEX error_dictionary_def_ix ON error_dictionary_def(short_desc);


CREATE TABLE run_comment_dat (
  iov_id		NUMBER(10),
  comment_id		NUMBER(10),
  source		varchar2(20),
  user_comment          varchar2(200),
  db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);

ALTER TABLE run_comment_dat ADD CONSTRAINT run_comment_dat_pk PRIMARY KEY (comment_id);
ALTER TABLE run_comment_dat ADD CONSTRAINT run_comment_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);
CREATE SEQUENCE run_comment_dat_SQ INCREMENT BY 1 START WITH 1;

CREATE trigger run_comment_dat_TRG
before insert on run_comment_dat
for each row
begin
select run_comment_dat_SQ.NextVal into :new.comment_id from dual;
end;
/

CREATE TABLE run_FEconfig_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10),
  config_id		NUMBER(10) 
);

ALTER TABLE run_FEconfig_dat ADD CONSTRAINT run_FEconfig_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE run_FEconfig_dat ADD CONSTRAINT run_FEconfig_dat_fk FOREIGN KEY (iov_id) REFERENCES run_iov (iov_id);
