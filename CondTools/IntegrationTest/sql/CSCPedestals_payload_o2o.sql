REM For CSCPEDESTALS:
insert into "CSCPEDESTALS"
SELECT
 record_id iov_value_id,
 run_num time
FROM PEDESTALS;

REM For CSCPEDESTALS_MAP:
insert into "CSCPEDESTALS_MAP"
SELECT
 map_index map_id,
 record_id iov_value_id,
 layer_id csc_int_id
FROM PEDESTALS_MAP
 order by iov_value_id,map_id;

REM For CSCPEDESTALS_DATA:
insert into "CSCPEDESTALS_DATA"
SELECT
 PEDESTALS_DATA.vec_index vec_index,
 PEDESTALS_MAP.map_index map_id,
 PEDESTALS_MAP.record_id iov_value_id,
 PEDESTALS_DATA.ped pedestals_ped, 
 PEDESTALS_DATA.rms pedestals_rms
FROM  PEDESTALS_DATA,PEDESTALS_MAP
WHERE
 PEDESTALS_DATA.map_id=PEDESTALS_MAP.map_id
ORDER BY
 iov_value_id,
 map_id,
 vec_index;
