/*
 *  Drops all tables created by create_laser_core.sql
 */

DROP TABLE lmf_run_iov;
DROP TABLE lmf_run_tag;

DROP SEQUENCE lmf_run_iov_sq;
DROP SEQUENCE lmf_run_tag_sq;

-- Done automatically:
-- DROP TRIGGER lmf_run_iov_tg;
