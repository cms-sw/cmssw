/* 
 * mon_summary.sql
 *
 * A program to output a summary of the monitoring data that has been
 * written to the online DB.  For every IOV and tag the number of
 * channels written to a data table is listed.
 */

set linesize 1000;
set pagesize 50000;
set feedback off;

alter session set nls_date_format='YYYY-MM-DD HH24:MI:SS';

DROP FUNCTION mon_summary;
DROP TYPE mon_summary_t;
DROP TYPE mon_summary_o;



CREATE OR replace TYPE mon_summary_o AS OBJECT (
  location varchar2(30),
  run_gen_tag varchar2(30),
  run_type varchar2(30),
  config_tag varchar2(30),
  config_ver number(10),
  run_iov_id number(10),
  run_num number(10),
  run_start date,
  run_end date,
  run_db_timestamp timestamp,
  mon_gen_tag varchar2(30),
  mon_ver varchar2(30),
  mon_iov_id number(10),
  subrun_num number(10),
  subrun_start date,
  subrun_end date,  
  mon_db_timestamp timestamp,
  
  crystal_consistency_cnt number(10),
  tt_consistency_cnt number(10),
  crystal_status_cnt number(10),
  pn_status_cnt number(10),
  occupancy_cnt number(10),
  pedestals_cnt number(10),
  pedestals_online_cnt number(10),
  pedestal_offsets_cnt number(10),
  test_pulse_cnt number(10),
  pulse_shape_cnt number(10),
  shape_quality_cnt number(10),
  delays_tt_cnt number(10),
  laser_blue_cnt number(10),
  laser_green_cnt number(10),
  laser_red_cnt number(10),
  laser_ired_cnt number(10),
  pn_blue_cnt number(10),
  pn_green_cnt number(10),
  pn_red_cnt number(10),
  pn_ired_cnt number(10),
  pn_ped_cnt number(10),
  pn_mgpa_cnt number(10),

  CONSTRUCTOR FUNCTION mon_summary_o(x NUMBER)
    RETURN SELF AS RESULT
);
/
show errors;



CREATE OR REPLACE TYPE BODY mon_summary_o AS
  CONSTRUCTOR FUNCTION mon_summary_o(x NUMBER)
    RETURN SELF AS RESULT
  AS
  BEGIN
  /* A null constructor */
    RETURN;
  END;
END;
/
show errors;



CREATE OR REPLACE TYPE mon_summary_t AS TABLE OF mon_summary_o;
/
show errors;



CREATE OR replace FUNCTION mon_summary (i_run NUMBER := NULL) RETURN mon_summary_t PIPELINED IS
  sql_stmt varchar2(4000);
  summary mon_summary_o := mon_summary_o(NULL);
  TYPE table_data_t IS TABLE OF NUMBER(10) INDEX BY VARCHAR2(32);
  table_data table_data_t;
  t VARCHAR2(32);  /* table name */
BEGIN

/* Initialize table_data */
table_data('MON_CRYSTAL_CONSISTENCY_DAT') := 0;
table_data('MON_TT_CONSISTENCY_DAT') := 0;
table_data('MON_CRYSTAL_STATUS_DAT') := 0;
table_data('MON_PN_STATUS_DAT') := 0;
table_data('MON_OCCUPANCY_DAT') := 0;
table_data('MON_PEDESTALS_DAT') := 0;
table_data('MON_PEDESTALS_ONLINE_DAT') := 0;
table_data('MON_PEDESTAL_OFFSETS_DAT') := 0;
table_data('MON_TEST_PULSE_DAT') := 0;
table_data('MON_PULSE_SHAPE_DAT') := 0;
table_data('MON_SHAPE_QUALITY_DAT') := 0;
table_data('MON_DELAYS_TT_DAT') := 0;
table_data('MON_LASER_BLUE_DAT') := 0;
table_data('MON_LASER_GREEN_DAT') := 0;
table_data('MON_LASER_RED_DAT') := 0;
table_data('MON_LASER_IRED_DAT') := 0;
table_data('MON_PN_BLUE_DAT') := 0;
table_data('MON_PN_GREEN_DAT') := 0;
table_data('MON_PN_RED_DAT') := 0;
table_data('MON_PN_IRED_DAT') := 0;
table_data('MON_PN_MGPA_DAT') := 0;

/* Loop through the MON runs */ 
FOR result IN (select loc.location, rtype.run_type, rtype.config_tag, rtype.config_ver, rtag.gen_tag run_gen_tag,
                      riov.iov_id run_iov_id, riov.run_num, riov.run_start, riov.run_end, riov.db_timestamp run_db_timestamp,
                      mver.mon_ver, mtag.gen_tag mon_gen_tag,
                      miov.iov_id mon_iov_id, miov.subrun_num, miov.subrun_start, miov.subrun_end, miov.db_timestamp mon_db_timestamp
                 from location_def loc 
                 join run_tag rtag on rtag.location_id = loc.def_id 
                 join run_type_def rtype on rtype.def_id = rtag.run_type_id
                 join run_iov riov on riov.tag_id = rtag.tag_id
                 join mon_run_iov miov on miov.run_iov_id = riov.iov_id
                 join mon_run_tag mtag on mtag.tag_id = miov.tag_id
                 join mon_version_def mver on mver.def_id = mtag.mon_ver_id
                 where riov.run_num = i_run
                 order by loc.location asc, 
                          riov.run_num asc, rtype.run_type asc, rtype.config_tag asc, rtag.gen_tag asc, 
                          miov.subrun_num asc, mver.mon_ver asc, mtag.gen_tag asc)
                          
LOOP
  summary.location         := result.location;
  summary.run_gen_tag      := result.run_gen_tag;
  summary.run_type         := result.run_type;
  summary.config_tag       := result.config_tag;
  summary.config_ver       := result.config_ver;
  summary.run_iov_id       := result.run_iov_id;
  summary.run_num          := result.run_num;
  summary.run_start        := result.run_start;
  summary.run_end          := result.run_end;
  summary.run_db_timestamp := result.run_db_timestamp;
  summary.mon_gen_tag      := result.mon_gen_tag;
  summary.mon_ver          := result.mon_ver;
  summary.mon_iov_id       := result.mon_iov_id;
  summary.subrun_num       := result.subrun_num;
  summary.subrun_start     := result.subrun_start;
  summary.subrun_end       := result.subrun_end;
  summary.mon_db_timestamp := result.mon_db_timestamp;

  /* Loop through all the data tables saving the number of channels written */
  t := table_data.FIRST;
  WHILE t IS NOT NULL
  LOOP
    sql_stmt := 'select count(*) from ' || t || ' where iov_id = :iov_id';
    EXECUTE IMMEDIATE sql_stmt INTO table_data(t) USING summary.mon_iov_id;
    t := table_data.NEXT(t);
  END LOOP;

  /* assign table_data to summary object elements */
  summary.crystal_consistency_cnt := table_data('MON_CRYSTAL_CONSISTENCY_DAT');
  summary.tt_consistency_cnt      := table_data('MON_TT_CONSISTENCY_DAT');
  summary.crystal_status_cnt      := table_data('MON_CRYSTAL_STATUS_DAT');
  summary.pn_status_cnt           := table_data('MON_PN_STATUS_DAT');
  summary.occupancy_cnt           := table_data('MON_OCCUPANCY_DAT');
  summary.pedestals_cnt           := table_data('MON_PEDESTALS_DAT');
  summary.pedestals_online_cnt    := table_data('MON_PEDESTALS_ONLINE_DAT');
  summary.pedestal_offsets_cnt    := table_data('MON_PEDESTAL_OFFSETS_DAT');
  summary.test_pulse_cnt          := table_data('MON_TEST_PULSE_DAT');
  summary.pulse_shape_cnt         := table_data('MON_PULSE_SHAPE_DAT');
  summary.shape_quality_cnt       := table_data('MON_SHAPE_QUALITY_DAT');
  summary.delays_tt_cnt           := table_data('MON_DELAYS_TT_DAT');
  summary.laser_blue_cnt          := table_data('MON_LASER_BLUE_DAT');
  summary.laser_green_cnt         := table_data('MON_LASER_GREEN_DAT');
  summary.laser_red_cnt           := table_data('MON_LASER_RED_DAT');
  summary.laser_ired_cnt          := table_data('MON_LASER_IRED_DAT');
  summary.pn_blue_cnt             := table_data('MON_PN_BLUE_DAT');
  summary.pn_green_cnt            := table_data('MON_PN_GREEN_DAT');
  summary.pn_red_cnt              := table_data('MON_PN_RED_DAT');
  summary.pn_ired_cnt             := table_data('MON_PN_IRED_DAT');
  summary.pn_ped_cnt              := table_data('MON_PN_IRED_DAT');
  summary.pn_mgpa_cnt             := table_data('MON_PN_MGPA_DAT');


  PIPE ROW(summary);
END LOOP;
RETURN;
END;
/
show errors;

/* Executes the summary program */
SELECT * FROM TABLE(mon_summary());
