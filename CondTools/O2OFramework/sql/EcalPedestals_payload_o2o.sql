/*
 *  EcalPedestals_payload_o2o()
 *
 *  EcalPedestals transform/transfer
 *  Parameters:  last_id:  The lower bounding IOV_VALUE_ID for objects to transfer
 */

CREATE OR REPLACE PROCEDURE EcalPedestals_payload_o2o (
  last_id IN NUMBER
)
AS

BEGIN
  INSERT INTO ecalpedestals
  (iov_value_id, time)
  SELECT 
    miov.iov_id, riov.run_num
  FROM 
    location_def@omds ldef, run_type_def@omds rdef, run_tag@omds rtag, run_iov@omds riov,
    /* Selects the mon_run_iov with the greatest subrun_num */
    (SELECT iov_id, run_iov_id, 
       MAX(subrun_num) KEEP (DENSE_RANK FIRST ORDER BY subrun_num ASC)
       FROM mon_run_iov@omds GROUP BY iov_id, run_iov_id) miov
  WHERE
      miov.run_iov_id = riov.iov_id
  AND riov.tag_id = rtag.tag_id
  AND rdef.def_id = rtag.run_type_id
  AND ldef.def_id = rtag.location_id
  AND ldef.location='P5_MT'
  AND rdef.run_type='PEDESTAL'
  AND miov.iov_id > last_id
;
    
  INSERT INTO ecalpedestals_item
  (iov_value_id, pos, det_id, mean_x12, rms_x12, mean_x6, rms_x6, mean_x1, rms_x1)
  SELECT 
    dat.iov_id,
    /* Creates a column of row numbers ordered by logic_id, for std::vec */
    ROW_NUMBER() OVER (PARTITION BY dat.iov_id ORDER BY dat.logic_id ASC),
    cv.id1,
    dat.ped_mean_g12,
    dat.ped_rms_g12,
    dat.ped_mean_g6,
    dat.ped_rms_g6,
    dat.ped_mean_g1,
    dat.ped_rms_g1
  FROM 
    ecalpedestals iov, mon_pedestals_dat@omds dat, channelview@omds cv
  WHERE dat.iov_id = iov.iov_value_id
    AND cv.logic_id = dat.logic_id
    AND cv.name='Offline_det_id'
    AND iov.iov_value_id > last_id
;
END;
/
show errors;
