/*
 *  Drops everything creates in create_run_core.sql
 */


DROP TABLE run_iov;
DROP TABLE run_tag;
DROP TABLE run_type_def;
DROP TABLE location_def;

DROP SEQUENCE location_def_sq;
DROP SEQUENCE run_type_def_sq;
DROP SEQUENCE run_iov_sq;
DROP SEQUENCE run_tag_sq;

-- Done automatically:
-- DROP TRIGGER run_iov_tg;
