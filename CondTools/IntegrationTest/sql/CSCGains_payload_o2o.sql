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

insert into "CSCGAINS"
SELECT
 record_id iov_value_id,
 run_num time
FROM GAINS@cmsomds
WHERE record_id > last_id
;

insert into "CSCGAINS_MAP"
SELECT
 map_index map_id,
 record_id iov_value_id,
 layer_id csc_int_id
FROM GAINS_MAP@cmsomds
WHERE record_id > last_id
;


insert into "CSCGAINS_DATA"
SELECT
 GAINS_DATA.vec_index vec_index,
 GAINS_MAP.map_index map_id,
 GAINS_MAP.record_id iov_value_id,
 GAINS_DATA.gain_chi2 gains_chi2, 
 GAINS_DATA.gain_intercept gains_intercept,
 GAINS_DATA.gain_slope gains_slope
FROM GAINS_DATA@cmsomds, GAINS_MAP@cmsomds
WHERE
 GAINS_DATA.map_id=GAINS_MAP.map_id
AND record_id > last_id
;

END;
/
show errors;
