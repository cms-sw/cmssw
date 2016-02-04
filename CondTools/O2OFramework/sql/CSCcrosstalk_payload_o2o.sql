/*
 *  CSCcrosstalk_payload_o2o()
 *
 *  CSCcrosstalk transform/transfer
 *  Parameters:  last_id:  The lower bounding IOV_VALUE_ID for objects to transfer
 */

CREATE OR REPLACE PROCEDURE CSCcrosstalk_payload_o2o (
  last_id IN NUMBER
)
AS

BEGIN

INSERT INTO "CSCCROSSTALK"
SELECT
 record_id iov_value_id,
 runs time
FROM crosstalk@omds
WHERE record_id > last_id and flag = 1
;

INSERT INTO "CSCCROSSTALK_MAP"
SELECT
 crosstalk_map.map_index map_id,
 crosstalk_map.record_id iov_value_id,
 crosstalk_map.layer_id csc_int_id
FROM crosstalk_map@omds, crosstalk@omds
WHERE crosstalk_map.record_id = crosstalk.record_id
  AND crosstalk_map.record_id > last_id
  AND crosstalk.flag = 1
;

INSERT INTO "CSCCROSSTALK_DATA"
SELECT
 crosstalk_data.vec_index vec_index,
 crosstalk_map.map_index map_id,
 crosstalk_map.record_id iov_value_id,
 crosstalk_data.xtalk_chi2_left crosstalk_chi2_left,
 crosstalk_data.xtalk_chi2_right crosstalk_chi2_right,
 crosstalk_data.xtalk_intercept_left crosstalk_intercept_left,
 crosstalk_data.xtalk_intercept_right crosstalk_intercept_right,
 crosstalk_data.xtalk_slope_left crosstalk_slope_left,
 crosstalk_data.xtalk_slope_right crosstalk_slope_right
FROM crosstalk_data@omds, crosstalk_map@omds, crosstalk@omds
WHERE crosstalk_data.map_id = crosstalk_map.map_id
  AND crosstalk_map.record_id = crosstalk.record_id
  AND crosstalk_map.record_id > last_id
  AND crosstalk.flag = 1
;

END;
/
show errors;
