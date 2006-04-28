/*
 *  CSCPedestals_payload_o2o()
 *
 *  CSCPedestals transform/transfer
 *  Parameters:  last_id:  The lower bounding IOV_VALUE_ID for objects to transfer
 */

CREATE OR REPLACE PROCEDURE CSCPedestals_payload_o2o (
  last_id IN NUMBER
)
AS

BEGIN

insert into "CSCPEDESTALS"
SELECT
 record_id iov_value_id,
 run_num time
FROM PEDESTALS@cmsomds
WHERE record_id > last_id
;


insert into "CSCPEDESTALS_MAP"
SELECT
 map_index map_id,
 record_id iov_value_id,
 layer_id csc_int_id
FROM PEDESTALS_MAP@cmsomds
WHERE record_id > last_id
;


insert into "CSCPEDESTALS_DATA"
SELECT
 PEDESTALS_DATA.vec_index vec_index,
 PEDESTALS_MAP.map_index map_id,
 PEDESTALS_MAP.record_id iov_value_id,
 PEDESTALS_DATA.ped pedestals_ped, 
 PEDESTALS_DATA.rms pedestals_rms
FROM PEDESTALS_DATA@cmsomds, PEDESTALS_MAP@cmsomds
WHERE
 PEDESTALS_DATA.map_id=PEDESTALS_MAP.map_id
AND PEDESTALS_MAP.record_id > last_id
;

END;
/
show errors;
