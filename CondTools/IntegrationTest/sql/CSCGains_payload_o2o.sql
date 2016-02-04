/*
 *  CSCGains_payload_o2o()
 *
 *  CSCGains transform/transfer
 *  Parameters:  last_id:  The lower bounding IOV_VALUE_ID for objects to transfer
 */

CREATE OR REPLACE PROCEDURE CSCGains_payload_o2o (
  last_id IN NUMBER
)
AS

BEGIN

INSERT INTO "CSCGAINS"
SELECT
 record_id iov_value_id,
 runs time
FROM gains@omds
WHERE record_id > last_id and flag = 1
;

INSERT INTO "CSCGAINS_MAP"
SELECT
 gains_map.map_index map_id,
 gains_map.record_id iov_value_id,
 gains_map.layer_id csc_int_id
FROM gains_map@omds, gains@omds
WHERE gains_map.record_id = gains.record_id
  AND gains_map.record_id > last_id 
  AND gains.flag = 1
;

INSERT INTO "CSCGAINS_DATA"
SELECT
 gains_data.vec_index vec_index,
 gains_map.map_index map_id,
 gains_map.record_id iov_value_id,
 gains_data.gain_chi2 gains_chi2, 
 gains_data.gain_intercept gains_intercept,
 gains_data.gain_slope gains_slope
FROM gains_data@omds, gains_map@omds, gains@omds
WHERE gains_data.map_id = gains_map.map_id
  AND gains_map.record_id = gains.record_id
  AND gains_map.record_id > last_id 
  AND gains.flag = 1
;

END;
/
show errors;
