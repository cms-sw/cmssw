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

insert into "CSCCROSSTALK"
SELECT
 record_id iov_value_id,
 run_num time
FROM CROSSTALK@omds
WHERE record_id > last_id
;


insert into "CSCCROSSTALK_MAP"
SELECT
 map_index map_id,
 record_id iov_value_id,
 layer_id csc_int_id
FROM CROSSTALK_MAP@omds
WHERE record_id > last_id
 order by iov_value_id,map_id;


insert into "CSCCROSSTALK_DATA"
SELECT
 CROSSTALK_DATA.vec_index vec_index,
 CROSSTALK_MAP.map_index map_id,
 CROSSTALK_MAP.record_id iov_value_id,
 CROSSTALK_DATA.xtalk_chi2_left crosstalk_chi2_left,
 CROSSTALK_DATA.xtalk_chi2_right crosstalk_chi2_right,
 CROSSTALK_DATA.xtalk_intercept_left crosstalk_intercept_left,
 CROSSTALK_DATA.xtalk_intercept_right crosstalk_intercept_right,
 CROSSTALK_DATA.xtalk_slope_left crosstalk_slope_left,
 CROSSTALK_DATA.xtalk_slope_right crosstalk_slope_right
FROM CROSSTALK_DATA@omds,CROSSTALK_MAP@omds
WHERE
 CROSSTALK_DATA.map_id=CROSSTALK_MAP.map_id
AND record_id > last_id
ORDER BY
 iov_value_id,
 map_id,
 vec_index;

END;
/
show errors;
