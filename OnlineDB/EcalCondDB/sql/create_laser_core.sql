/*
 * Creates the def, tag, and iov tables necessary to define a laser monitoring farm subrun
 * Requires: create_run_core.sql
 */



/* laser monitoring farm tag */
CREATE TABLE lmf_run_tag (
  tag_id		NUMBER(10) NOT NULL,
  gen_tag		VARCHAR2(100) NOT NULL
);

CREATE SEQUENCE lmf_run_tag_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE lmf_run_tag ADD CONSTRAINT lmf_run_tag_pk PRIMARY KEY (tag_id);
ALTER TABLE lmf_run_tag ADD CONSTRAINT lmf_run_tag_uk UNIQUE (gen_tag);



/* laser monitoring farm IOV */
CREATE TABLE lmf_run_iov (
  iov_id		NUMBER(10) NOT NULL,
  tag_id		NUMBER(10) NOT NULL,
  run_iov_id		NUMBER(10) NOT NULL,
  subrun_num		NUMBER(10) NOT NULL,
  subrun_start		DATE NOT NULL,
  subrun_end		DATE NOT NULL,
  db_timestamp		TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE SEQUENCE lmf_run_iov_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE lmf_run_iov ADD CONSTRAINT lmf_run_iov_pk PRIMARY KEY (iov_id);
ALTER TABLE lmf_run_iov ADD CONSTRAINT lmf_run_iov_uk UNIQUE (run_iov_id, subrun_num);
CREATE INDEX lmf_run_iov_ix ON lmf_run_iov (subrun_start, subrun_end);
ALTER TABLE lmf_run_iov ADD CONSTRAINT lmf_run_iov_fk1 FOREIGN KEY (tag_id) REFERENCES lmf_run_tag (tag_id);
ALTER TABLE lmf_run_iov ADD CONSTRAINT lmf_run_iov_fk2 FOREIGN KEY (run_iov_id) REFERENCES run_iov (iov_id);



/* laser monitoring farm triggers, constraint checks */
CREATE OR REPLACE TRIGGER lmf_run_iov_tg
  BEFORE INSERT ON lmf_run_iov
  REFERENCING NEW AS newiov
  FOR EACH ROW
  CALL update_subrun_iov_dates('lmf_run_iov', 'subrun_start', 'subrun_end', :newiov.subrun_start, :newiov.subrun_end, :newiov.tag_id, :newiov.run_iov_id)
/
