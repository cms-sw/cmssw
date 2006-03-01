/*
 *  Drops all tables created by create_monitoring_core.sql
 */

DROP TABLE mon_run_iov;
DROP TABLE mon_run_tag;
DROP TABLE mon_version_def;

DROP SEQUENCE mon_run_iov_sq;
DROP SEQUENCE mon_run_tag_sq;
DROP SEQUENCE mon_version_def_sq;

-- Done automatically:
-- DROP TRIGGER mon_run_iov_tg;
