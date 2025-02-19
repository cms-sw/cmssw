SET ECHO OFF;
SET FEEDBACK OFF;
SET LINESIZE 1000;
SET PAGESIZE 1000;
col location format a8
col gen_tag format a15
col run_type format a15
col config_tag format a15
col mon_ver format a15

ALTER SESSION SET NLS_DATE_FORMAT='YYYY-MM-DD HH24:MI:SS';

SELECT ri.run_num, mi.subrun_num, d.num_chan, ld.location, rt.gen_tag, rd.run_type, rd.config_tag, rd.config_ver, mvd.mon_ver,
       ri.run_start, ri.run_end, mi.subrun_start, mi.subrun_end, mi.db_timestamp
FROM 
(SELECT iov_id, count(logic_id) num_chan FROM mon_pedestals_dat GROUP BY iov_id) d
JOIN mon_run_iov mi ON d.iov_id = mi.iov_id
JOIN mon_run_tag mt ON mt.tag_id = mi.tag_id
JOIN mon_version_def mvd ON mvd.def_id = mt.mon_ver_id
JOIN run_iov ri ON ri.iov_id = mi.run_iov_id
JOIN run_tag rt ON rt.tag_id = ri.tag_id
JOIN run_type_def rd ON rd.def_id = rt.run_type_id
JOIN location_def ld ON ld.def_id = rt.location_id
-- WHERE rownum < 50
ORDER BY ri.run_num, mi.subrun_num;
