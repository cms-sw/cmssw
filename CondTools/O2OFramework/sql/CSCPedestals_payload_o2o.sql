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

INSERT INTO "CSCPEDESTALS"
SELECT
 record_id iov_value_id,
 runs time
FROM pedestals@omds
WHERE record_id > last_id and flag = 1
;


INSERT INTO "CSCPEDESTALS_MAP"
SELECT
 pedestals_map.map_index map_id,
 pedestals_map.record_id iov_value_id,
 pedestals_map.layer_id csc_int_id
FROM pedestals_map@omds, pedestals@omds
WHERE pedestals.record_id = pedestals_map.record_id
  AND pedestals_map.record_id > last_id 
  AND pedestals.flag = 1
;


INSERT INTO "CSCPEDESTALS_DATA"
SELECT
 pedestals_data.vec_index vec_index,
 pedestals_map.map_index map_id,
 pedestals_map.record_id iov_value_id,
 pedestals_data.ped pedestals_ped, 
 pedestals_data.rms pedestals_rms
FROM pedestals_data@omds, pedestals_map@omds, pedestals@omds
WHERE pedestals_data.map_id=pedestals_map.map_id
  AND pedestals_map.record_id = pedestals.record_id
  AND pedestals_map.record_id > last_id 
  AND pedestals.flag = 1
;

END;
/
show errors;
