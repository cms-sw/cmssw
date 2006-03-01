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
