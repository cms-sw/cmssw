/*
 * Creates the def, tag and iov tables needed to define an ECAL run
 */



/* location definition table */
CREATE TABLE location_def (
  def_id		NUMBER NOT NULL,
  location		VARCHAR2(100) NOT NULL
);

CREATE SEQUENCE location_def_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE location_def ADD CONSTRAINT location_def_pk PRIMARY KEY (def_id);
ALTER TABLE location_def ADD CONSTRAINT location_def_uk UNIQUE (location);



/* run type definition table */
CREATE TABLE run_type_def (
  def_id		NUMBER(10) NOT NULL,
  run_type		VARCHAR2(100) NOT NULL,
  description		VARCHAR2(1000) NOT NULL
);

CREATE SEQUENCE run_type_def_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE run_type_def ADD CONSTRAINT run_type_def_pk PRIMARY KEY (def_id);
ALTER TABLE run_type_def ADD CONSTRAINT run_type_def_uk UNIQUE (run_type);



/* run tag*/
CREATE TABLE run_tag (
  tag_id		NUMBER(10) NOT NULL,
  gen_tag		VARCHAR(100) NOT NULL,
  location_id		NUMBER(10) NOT NULL,
  run_type_id		NUMBER(10) NOT NULL
);

CREATE SEQUENCE run_tag_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE run_tag ADD CONSTRAINT run_tag_pk PRIMARY KEY (tag_id);
ALTER TABLE run_tag ADD CONSTRAINT run_tag_uk UNIQUE (gen_tag, location_id, run_type_id);
ALTER TABLE run_tag ADD CONSTRAINT run_tag_fk1 FOREIGN KEY (location_id) REFERENCES location_def (def_id);
ALTER TABLE run_tag ADD CONSTRAINT run_tag_fk2 FOREIGN KEY (run_type_id) REFERENCES run_type_def (def_id);



/* run iov */
CREATE TABLE run_iov (
  iov_id		NUMBER(10) NOT NULL,
  tag_id		NUMBER(10) NOT NULL,
  run_num		NUMBER(10) NOT NULL,
  run_start		DATE NOT NULL,
  run_end		DATE NOT NULL,
  db_timestamp		TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE SEQUENCE run_iov_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE run_iov ADD CONSTRAINT run_iov_pk PRIMARY KEY (iov_id);
ALTER TABLE run_iov ADD CONSTRAINT run_iov_uk UNIQUE (tag_id, run_num);
ALTER TABLE run_iov ADD CONSTRAINT run_iov_uk2 UNIQUE (tag_id, run_start);
CREATE INDEX run_iov_ix ON run_iov (run_start, run_end);
ALTER TABLE run_iov ADD CONSTRAINT run_iov_fk FOREIGN KEY (tag_id) REFERENCES run_tag (tag_id);



/* triggers, constraint checks */
CREATE OR REPLACE TRIGGER run_iov_tg
  BEFORE INSERT ON run_iov
  REFERENCING NEW AS newiov
  FOR EACH ROW
  CALL update_iov_dates('run_iov', 'run_start', 'run_end', :newiov.run_start, :newiov.run_end, :newiov.tag_id)
/
