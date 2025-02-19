/* 
 * dcu_summary.sql
 *
 * A program to output a summary of the DCU data that has been
 * written to the online DB.  For every IOV and tag the number of
 * channels written to a data table is listed.
 */


set linesize 1000;
set pagesize 50000;
set feedback off;

alter session set nls_date_format='YYYY-MM-DD HH24:MI:SS';

DROP FUNCTION dcu_summary;
DROP TYPE dcu_summary_t;
DROP TYPE dcu_summary_o;



CREATE OR replace TYPE dcu_summary_o AS OBJECT (
  location varchar2(30),
  gen_tag varchar2(30),
  since date,
  till date,
  iov_id number(10),
  db_timestamp timestamp,
  capsule_temp_cnt number(10),
  capsule_temp_raw_cnt number(10),
  idark_cnt number(10),
  idark_ped_cnt number(10),
  vfe_temp_cnt number(10),
  lvr_temps_cnt number(10),
  lvrb_temps_cnt number(10),
  lvr_voltages_cnt number(10),
  
  CONSTRUCTOR FUNCTION dcu_summary_o(x NUMBER)
    RETURN SELF AS RESULT
);
/
show errors;



CREATE OR REPLACE TYPE BODY dcu_summary_o AS
  CONSTRUCTOR FUNCTION dcu_summary_o(x NUMBER)
    RETURN SELF AS RESULT
  AS
  BEGIN
  /* A null constructor */
    RETURN;
  END;
END;
/
show errors;



CREATE OR REPLACE TYPE dcu_summary_t AS TABLE OF dcu_summary_o;
/
show errors;



CREATE OR replace FUNCTION dcu_summary RETURN dcu_summary_t PIPELINED IS
  sql_stmt varchar2(4000);
  summary dcu_summary_o := dcu_summary_o(NULL);
  TYPE table_data_t IS TABLE OF NUMBER(10) INDEX BY VARCHAR2(32);
  table_data table_data_t;
  t VARCHAR2(32);  /* table name */
BEGIN

/* Initialize table_data */
table_data('DCU_CAPSULE_TEMP_DAT') := 0;
table_data('DCU_CAPSULE_TEMP_RAW_DAT') := 0;
table_data('DCU_IDARK_DAT') := 0;
table_data('DCU_IDARK_PED_DAT') := 0;
table_data('DCU_VFE_TEMP_DAT') := 0;
table_data('DCU_LVR_TEMPS_DAT') := 0;
table_data('DCU_LVRB_TEMPS_DAT') := 0;
table_data('DCU_LVR_VOLTAGES_DAT') := 0;

/* Loop through the DCU runs */ 
FOR result IN (select loc.location, tag.gen_tag, iov.since, iov.till, iov.iov_id, iov.db_timestamp
                 from location_def loc 
                 join dcu_tag tag on tag.location_id = loc.def_id 
                 join dcu_iov iov on iov.tag_id=tag.tag_id
                 order by loc.location asc, tag.gen_tag asc, iov.since asc)
LOOP
  summary.location := result.location;
  summary.gen_tag := result.gen_tag;
  summary.since := result.since;
  summary.till := result.till;  
  summary.iov_id := result.iov_id;
  summary.db_timestamp := result.db_timestamp;

  /* Loop through all the data tables saving the number of channels written */
  t := table_data.FIRST;
  WHILE t IS NOT NULL
  LOOP
    sql_stmt := 'select count(*) from ' || t || ' where iov_id = :iov_id';
    EXECUTE IMMEDIATE sql_stmt INTO table_data(t) USING summary.iov_id;
    t := table_data.NEXT(t);
  END LOOP;

  /* assign table_data to summary object elements */
  summary.capsule_temp_cnt     := table_data('DCU_CAPSULE_TEMP_DAT');
  summary.capsule_temp_raw_cnt := table_data('DCU_CAPSULE_TEMP_RAW_DAT');
  summary.idark_cnt            := table_data('DCU_IDARK_DAT');
  summary.idark_ped_cnt        := table_data('DCU_IDARK_PED_DAT');
  summary.vfe_temp_cnt         := table_data('DCU_VFE_TEMP_DAT');
  summary.lvrb_temps_cnt       := table_data('DCU_LVRB_TEMPS_DAT');
  summary.lvr_temps_cnt        := table_data('DCU_LVR_TEMPS_DAT');
  summary.lvr_voltages_cnt     := table_data('DCU_LVR_VOLTAGES_DAT');

  PIPE ROW(summary);
END LOOP;
RETURN;
END;
/
show errors;

/* Executes the summary program */
SELECT * FROM TABLE(dcu_summary());
