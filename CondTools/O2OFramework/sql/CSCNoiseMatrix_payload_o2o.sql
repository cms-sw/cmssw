/*
 *  CSCNoiseMatrix_payload_o2o()
 *
 *  CSCNoiseMatrix transform/transfer
 *  Parameters:  last_id:  The lower bounding IOV_VALUE_ID for objects to transfer
 */

CREATE OR REPLACE PROCEDURE CSCNoiseMatrix_payload_o2o (
  last_id IN NUMBER
)
AS

BEGIN

INSERT INTO "CSCNOISEMATRIX"
SELECT
 record_id iov_value_id,
 runs time
FROM noisematrix@omds
WHERE record_id > last_id and flag = 1
;

INSERT INTO "CSCNOISEMATRIX_MAP"
SELECT
 noisematrix_map.map_index map_id,
 noisematrix_map.record_id iov_value_id,
 noisematrix_map.layer_id csc_int_id
FROM noisematrix_map@omds, noisematrix@omds
WHERE noisematrix_map.record_id = noisematrix.record_id
  AND noisematrix_map.record_id > last_id
  AND noisematrix.flag = 1
;

INSERT INTO "CSCNOISEMATRIX_DATA"
SELECT
 noisematrix_data.vec_index vec_index,
 noisematrix_map.map_index map_id,
 noisematrix_map.record_id iov_value_id,
 noisematrix_data.elem33 cscmatrix_elem33,
 noisematrix_data.elem34 cscmatrix_elem34,
 noisematrix_data.elem35 cscmatrix_elem35,
 noisematrix_data.elem44 cscmatrix_elem44,
 noisematrix_data.elem45 cscmatrix_elem45,
 noisematrix_data.elem46 cscmatrix_elem46,
 noisematrix_data.elem55 cscmatrix_elem55,
 noisematrix_data.elem56 cscmatrix_elem56,
 noisematrix_data.elem57 cscmatrix_elem57,
 noisematrix_data.elem66 cscmatrix_elem66,
 noisematrix_data.elem67 cscmatrix_elem67,
 noisematrix_data.elem77 cscmatrix_elem77
FROM noisematrix_data@omds, noisematrix_map@omds, noisematrix@omds
WHERE noisematrix_data.map_id = noisematrix_map.map_id
  AND noisematrix_map.record_id = noisematrix.record_id
  AND noisematrix.record_id > last_id
  AND noisematrix.flag = 1
;

END;
/
show errors;
