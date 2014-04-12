/*
 * Creates the def, tag, and iov tables necessary to define a monitoring subrun
 * Requires: create_run_core.sql
 */

/* monitoring version definition */
CREATE TABLE mon_version_def (
  def_id		NUMBER(10) NOT NULL,
  mon_ver		VARCHAR2(100) NOT NULL,
  description		VARCHAR2(1000) NOT NULL
);

CREATE SEQUENCE mon_version_def_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE mon_version_def ADD CONSTRAINT mon_version_def_pk PRIMARY KEY (def_id);
ALTER TABLE mon_version_def ADD CONSTRAINT mon_version_def_uk UNIQUE (mon_ver);



/* monitoring tag */
CREATE TABLE mon_run_tag (
  tag_id		NUMBER(10) NOT NULL,
  gen_tag		VARCHAR2(100) NOT NULL,
  mon_ver_id		NUMBER(10) NOT NULL
);

CREATE SEQUENCE mon_run_tag_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE mon_run_tag ADD CONSTRAINT mon_run_tag_pk PRIMARY KEY (tag_id);
ALTER TABLE mon_run_tag ADD CONSTRAINT mon_run_tag_uk UNIQUE (gen_tag, mon_ver_id);
ALTER TABLE mon_run_tag ADD CONSTRAINT mon_run_tag_fk FOREIGN KEY (mon_ver_id) REFERENCES mon_version_def (def_id);



/* monitoring IOV */
CREATE TABLE mon_run_iov (
  iov_id		NUMBER(10) NOT NULL,
  tag_id		NUMBER(10) NOT NULL,
  run_iov_id		NUMBER(10) NOT NULL,
  subrun_num		NUMBER(10) NOT NULL,
  subrun_start		DATE NOT NULL,
  subrun_end		DATE NOT NULL,
  db_timestamp		TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE SEQUENCE mon_run_iov_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE mon_run_iov ADD CONSTRAINT mon_run_iov_pk PRIMARY KEY (iov_id);
ALTER TABLE mon_run_iov ADD CONSTRAINT mon_run_iov_uk UNIQUE (tag_id, run_iov_id, subrun_num);
CREATE INDEX mon_run_iov_ix ON mon_run_iov (subrun_start, subrun_end);
ALTER TABLE mon_run_iov ADD CONSTRAINT mon_run_iov_fk1 FOREIGN KEY (tag_id) REFERENCES mon_run_tag (tag_id);
ALTER TABLE mon_run_iov ADD CONSTRAINT mon_run_iov_fk2 FOREIGN KEY (run_iov_id) REFERENCES run_iov (iov_id);



/* monitoring triggers, constraint checks */
CREATE OR REPLACE TRIGGER mon_run_iov_tg
  BEFORE INSERT ON mon_run_iov
  REFERENCING NEW AS newiov
  FOR EACH ROW
  CALL update_subrun_iov_dates('mon_run_iov', 'subrun_start', 'subrun_end', :newiov.subrun_start, :newiov.subrun_end, :newiov.tag_id, :newiov.run_iov_id)
/
SHOW ERRORS;


CREATE TABLE mon_task_def (
  def_id		NUMBER(10) NOT NULL,
  task_code		VARCHAR2(3) NOT NULL,
  task_bit		number(10) NOT NULL,
  task_name             varchar2(30) not null,
  task_description      varchar2(100) not null
);

CREATE SEQUENCE mon_task_def_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE mon_task_def ADD CONSTRAINT mon_task_def_pk PRIMARY KEY (def_id);
ALTER TABLE mon_task_def ADD CONSTRAINT mon_task_def_uk UNIQUE (task_code);
ALTER TABLE mon_task_def ADD CONSTRAINT mon_task_def_uk2 UNIQUE (task_bit);
