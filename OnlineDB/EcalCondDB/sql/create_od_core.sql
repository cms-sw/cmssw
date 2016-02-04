/* Off Detector electronics monitoring IOV */
CREATE TABLE od_run_iov (
  iov_id		NUMBER(10) NOT NULL,
  run_iov_id		NUMBER(10) NOT NULL,
  subrun_num		NUMBER(10) NOT NULL,
  subrun_start		DATE NOT NULL,
  subrun_end		DATE NOT NULL,
  db_timestamp		TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

CREATE SEQUENCE od_run_iov_sq INCREMENT BY 1 START WITH 1;

ALTER TABLE od_run_iov ADD CONSTRAINT od_run_iov_pk PRIMARY KEY (iov_id);
ALTER TABLE od_run_iov ADD CONSTRAINT od_run_iov_uk UNIQUE (run_iov_id, subrun_num);
CREATE INDEX od_run_iov_ix ON od_run_iov (subrun_start, subrun_end);
ALTER TABLE od_run_iov ADD CONSTRAINT od_run_iov_fk1 FOREIGN KEY (run_iov_id) REFERENCES run_iov (iov_id);




/* monitoring triggers, constraint checks */
CREATE OR REPLACE TRIGGER od_run_iov_tg
  BEFORE INSERT ON od_run_iov
  REFERENCING NEW AS newiov
  FOR EACH ROW
  CALL update_odsubrun_iov_dates('od_run_iov', 'subrun_start', 'subrun_end', :newiov.subrun_start, :newiov.subrun_end, :newiov.run_iov_id)
/
SHOW ERRORS;








