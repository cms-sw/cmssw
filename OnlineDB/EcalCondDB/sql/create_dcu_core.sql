/*
 * Creates the def, tag and iov tables needed to define a DCU data taking
 */




/* dcu tag*/
CREATE TABLE dcu_tag (
  tag_id		NUMBER(10) NOT NULL,
  gen_tag		VARCHAR(100) NOT NULL,
  location_id		NUMBER(10) NOT NULL
);

CREATE SEQUENCE dcu_tag_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE dcu_tag ADD CONSTRAINT dcu_tag_pk PRIMARY KEY (tag_id);
ALTER TABLE dcu_tag ADD CONSTRAINT dcu_tag_uk UNIQUE (gen_tag, location_id);
ALTER TABLE dcu_tag ADD CONSTRAINT dcu_tag_fk1 FOREIGN KEY (location_id) REFERENCES location_def (def_id);



/* dcu iov */
CREATE TABLE dcu_iov (
  iov_id		NUMBER(10) NOT NULL,
  tag_id		NUMBER(10) NOT NULL,
  since			DATE NOT NULL,
  till			DATE NOT NULL,
  db_timestamp		TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE SEQUENCE dcu_iov_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE dcu_iov ADD CONSTRAINT dcu_iov_pk PRIMARY KEY (iov_id);
CREATE INDEX dcu_iov_ix ON dcu_iov (since, till);
ALTER TABLE dcu_iov ADD CONSTRAINT dcu_iov_fk FOREIGN KEY (tag_id) REFERENCES dcu_tag (tag_id);



/* triggers, constraint checks */
CREATE OR REPLACE TRIGGER dcu_iov_tg
  BEFORE INSERT ON dcu_iov
  REFERENCING NEW AS newiov
  FOR EACH ROW
  CALL update_iov_dates('dcu_iov', 'since', 'till', :newiov.since, :newiov.till, :newiov.tag_id)
/
