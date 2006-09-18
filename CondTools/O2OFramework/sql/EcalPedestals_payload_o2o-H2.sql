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
    /* Selects the mon_run_iov with the greatest subrun_num */
    (SELECT iov_id, run_iov_id, 
       MAX(subrun_num) KEEP (DENSE_RANK FIRST ORDER BY subrun_num ASC)
       FROM mon_run_iov@ecalh4db GROUP BY iov_id, run_iov_id) miov
    INNER JOIN run_iov@ecalh4db riov
      ON miov.run_iov_id = riov.iov_id
    INNER JOIN run_tag@ecalh4db rtag
      ON riov.tag_id = rtag.tag_id
    INNER JOIN location_def@ecalh4db ldef
      ON rtag.location_id = ldef.def_id
    INNER JOIN run_type_def@ecalh4db rdef
      ON rtag.run_type_id = rdef.def_id    
  WHERE
      ldef.location='H2'
  AND rdef.run_type='PEDESTAL'
  AND miov.iov_id > last_id
;
    
  /* The following query has been modified to include a horrible 3-time self-join of the channelview 
     table with the purpose of remapping the data from SM1 to SM4 as required for ECAL-HCAL alignment 
     cv1 maps the data to its native mapping
     cv2 maps the native mapping (cv1) to remap SM1 to SM4
     cv3 maps the altered mapping (cv2) to the det_id
     
     The FROM block from above is repeated here, instead of using the
     results already transferred to the ECALPEDESTALS table.  This is
     because using a combination of the local tables and remote tables
     has /very/ poor performance.
   */
  INSERT INTO ecalpedestals_item
  (iov_value_id, pos, det_id, mean_x12, rms_x12, mean_x6, rms_x6, mean_x1, rms_x1)
  SELECT 
    dat.iov_id,
    /* Creates a column of row numbers ordered by logic_id, for std::vec */
    ROW_NUMBER() OVER (PARTITION BY dat.iov_id ORDER BY dat.logic_id ASC),
    cv3.id1,
    dat.ped_mean_g12,
    dat.ped_rms_g12,
    dat.ped_mean_g6,
    dat.ped_rms_g6,
    dat.ped_mean_g1,
    dat.ped_rms_g1
  FROM 
    /* Begin IOV selection block */
    /* Selects the mon_run_iov with the greatest subrun_num */
    (SELECT iov_id, run_iov_id, 
       MAX(subrun_num) KEEP (DENSE_RANK FIRST ORDER BY subrun_num ASC)
       FROM mon_run_iov@ecalh4db GROUP BY iov_id, run_iov_id) miov
    INNER JOIN run_iov@ecalh4db riov
      ON miov.run_iov_id = riov.iov_id
    INNER JOIN run_tag@ecalh4db rtag
      ON riov.tag_id = rtag.tag_id
    INNER JOIN location_def@ecalh4db ldef
      ON rtag.location_id = ldef.def_id
    INNER JOIN run_type_def@ecalh4db rdef
      ON rtag.run_type_id = rdef.def_id
    /* End IOV selection block */
    INNER JOIN mon_pedestals_dat@ecalh4db dat
      ON miov.iov_id = dat.iov_id
    INNER JOIN channelview@ecalh4db cv1
      ON dat.logic_id = cv1.logic_id AND cv1.name = 'EB_crystal_number' AND cv1.maps_to = 'EB_crystal_number'
    INNER JOIN channelview@ecalh4db cv2
      ON cv1.id2 = cv2.id2 AND cv2.id1 = 4 AND cv2.name = 'EB_crystal_number' AND cv2.maps_to = 'EB_crystal_number'
    INNER JOIN channelview@ecalh4db cv3
      ON cv2.logic_id = cv3.logic_id AND cv3.name = 'Offline_det_id' AND cv3.maps_to = 'EB_crystal_number'
  WHERE 
      ldef.location='H2'
  AND rdef.run_type='PEDESTAL'
  AND miov.iov_id > last_id
;
END;
/
show errors;
