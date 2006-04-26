REM For CSCGAINS:
insert into "CSCGAINS"
SELECT
 record_id iov_value_id,
 run_num time
FROM GAINS;

REM For CSCGAINS_MAP:
insert into "CSCGAINS_MAP"
SELECT
 map_index map_id,
 record_id iov_value_id,
 layer_id csc_int_id
FROM GAINS_MAP
 order by iov_value_id,map_id;

REM For CSCGAINS_DATA:
insert into "CSCGAINS_DATA"
SELECT
 GAINS_DATA.vec_index vec_index,
 GAINS_MAP.map_index map_id,
 GAINS_MAP.record_id iov_value_id,
 GAINS_DATA.gain_chi2 gains_chi2, 
 GAINS_DATA.gain_intercept gains_intercept,
 GAINS_DATA.gain_slope gains_slope
FROM GAINS_DATA,GAINS_MAP
WHERE
 GAINS_DATA.map_id=GAINS_MAP.map_id
ORDER BY
 iov_value_id,
 map_id,
 vec_index;
