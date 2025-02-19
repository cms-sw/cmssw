/*
 *  Creates all the data tables referencing od_run_iov
 *  Requires:  create_od_core.sql
 */



CREATE TABLE od_ccs_tr_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- token ring
  ccs_word		integer
);

ALTER TABLE od_ccs_tr_dat ADD CONSTRAINT od_ccs_tr_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE od_ccs_tr_dat ADD CONSTRAINT od_ccs_tr_dat_fk1 FOREIGN KEY (iov_id) REFERENCES od_run_iov (iov_id);

CREATE TABLE od_ccs_fe_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- fe
  ccs_word		integer
);

ALTER TABLE od_ccs_fe_dat ADD CONSTRAINT od_ccs_fe_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE od_ccs_fe_dat ADD CONSTRAINT od_ccs_fe_dat_fk1 FOREIGN KEY (iov_id) REFERENCES od_run_iov (iov_id);


create table od_ccs_hf_dat ( 
   iov_id   NUMBER(10),
  logic_id              NUMBER(10), 
  ccs_log         CLOB
);

ALTER TABLE od_ccs_hf_dat ADD CONSTRAINT od_ccs_hf_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE od_ccs_hf_dat ADD CONSTRAINT od_ccs_hf_dat_fk1 FOREIGN KEY (iov_id) REFERENCES od_run_iov (iov_id);

CREATE TABLE od_dcc_operation_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- dcc
operationMode		varchar2(40)
);

ALTER TABLE od_dcc_operation_dat ADD CONSTRAINT od_dcc_operation_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE od_dcc_operation_dat ADD CONSTRAINT od_dcc_operation_dat_fk1 FOREIGN KEY (iov_id) REFERENCES od_run_iov (iov_id);

CREATE TABLE od_dcc_details_dat (
  iov_id		NUMBER(10),
  logic_id		NUMBER(10), -- dcc channel
  qpll_error		NUMBER(10),
  optical_link		NUMBER(10),
  data_timeout		NUMBER(10),
  dcc_header		NUMBER(10),
  event_number		NUMBER(10),
  bx_number		NUMBER(10),
  even_parity		NUMBER(10),
  odd_parity		NUMBER(10),
  block_size		NUMBER(10),
  almost_full_fifo	NUMBER(10),
  full_fifo		NUMBER(10),
  forced_full_supp	NUMBER(10)
);

ALTER TABLE od_dcc_details_dat ADD CONSTRAINT od_dcc_details_dat_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE od_dcc_details_dat ADD CONSTRAINT od_dcc_details_dat_fk1 FOREIGN KEY (iov_id) REFERENCES od_run_iov (iov_id);
