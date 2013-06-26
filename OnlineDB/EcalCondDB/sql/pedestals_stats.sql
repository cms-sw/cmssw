SET ECHO OFF;
SET FEEDBACK OFF;
SET LINESIZE 1000;
SET PAGESIZE 1000;
col location format a8
col gen_tag format a8
col run_type format a15
col monitoring_ver format a15

ALTER SESSION SET NLS_DATE_FORMAT='YYYY-MM-DD HH24:MI:SS';
SELECT location, gen_tag, run_type, monitoring_ver,
       count(distinct run_num) num_runs,
       min(run_num) min_run, min(run_start) min_start_date,
       max(run_num) max_run, max(run_start) max_start_date
FROM (
  SELECT lt.location, rt.tag_id, rt.gen_tag, rt.run_type, rt.monitoring_ver,
         i.run_num, i.run_start, i.run_end
  FROM location_tag lt JOIN run_tag rt ON lt.tag_id = rt.location_id
  JOIN run_iov i ON i.tag_id = rt.tag_id
  JOIN (SELECT DISTINCT iov_id FROM mon_pedestals_dat) m ON m.iov_id = i.iov_id
)
GROUP BY location, gen_tag, run_type, monitoring_ver;
