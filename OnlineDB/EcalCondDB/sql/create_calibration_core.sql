/*
 * Creates the def, tag and iov tables needed to define a set of calibrations
 */

/*  cali_tag */
CREATE TABLE cali_tag (
  tag_id		NUMBER(10) NOT NULL,
  location_id		NUMBER(10) NOT NULL,
  gen_tag		VARCHAR(30) NOT NULL,
  method		VARCHAR(40) NOT NULL,
  version		VARCHAR(40) NOT NULL,
  data_type		VARCHAR(40) NOT NULL
);

CREATE SEQUENCE cali_tag_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE cali_tag ADD CONSTRAINT cali_tag_pk PRIMARY KEY (tag_id);
ALTER TABLE cali_tag ADD CONSTRAINT cali_tag_fk FOREIGN KEY (location_id) REFERENCES location_def (def_id);

/* cali iov */
CREATE TABLE cali_iov (
  iov_id		NUMBER(10) NOT NULL,
  tag_id		NUMBER(10) NOT NULL,
  since			DATE NOT NULL,
  till			DATE NOT NULL,
  db_timestamp		TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE SEQUENCE cali_iov_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE cali_iov ADD CONSTRAINT cali_iov_pk PRIMARY KEY (iov_id);
CREATE INDEX cali_iov_ix ON cali_iov(since, till);
ALTER TABLE cali_iov ADD CONSTRAINT cali_iov_fk FOREIGN KEY (tag_id) REFERENCES cali_tag (tag_id);



/* triggers, constraint checks */
CREATE OR REPLACE TRIGGER cali_iov_tg
  BEFORE INSERT ON cali_iov
  REFERENCING NEW AS newiov
  FOR EACH ROW
  CALL update_iov_dates('cali_iov', 'since', 'till', :newiov.since, :newiov.till, :newiov.tag_id)
/
