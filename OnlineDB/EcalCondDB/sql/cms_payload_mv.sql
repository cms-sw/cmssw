CREATE MATERIALIZED VIEW ECALPEDESTALS 
AS 
/* query to fill ECALPEDESTALS table */
  SELECT 
    miov.iov_id iov_value_id, 
    riov.run_num time
  FROM 
    /* inline view selects the mon_run_iov with the greatest subrun_num */
    (SELECT iov_id, run_iov_id, 
       MAX(subrun_num) KEEP (DENSE_RANK FIRST ORDER BY subrun_num ASC)
       FROM mon_run_iov@cmsomds GROUP BY iov_id, run_iov_id) miov
    JOIN run_iov@cmsomds riov ON miov.run_iov_id = riov.iov_id
    JOIN run_tag@cmsomds rtag ON riov.tag_id = rtag.tag_id
    JOIN run_type_def@cmsomds rdef ON rtag.run_type_id = rdef.def_id
    JOIN location_def@cmsomds ldef ON rtag.location_id = ldef.def_id
  WHERE
      ldef.location='H4'
  AND rdef.run_type='PEDESTAL'
  AND rdef.config_tag='PEDESTAL-STD'
  AND rdef.config_ver=1
;

/* query to fill ECALPEDESTALS_ITEM table */
CREATE MATERIALIZED VIEW ECALPEDESTALS_ITEM
AS
  SELECT 
    dat.iov_id iov_value_id,
    /* Creates a column of row numbers ordered by logic_id, for std::vec */
    ROW_NUMBER() OVER (PARTITION BY dat.iov_id ORDER BY dat.logic_id ASC) pos,
    cv.id1 det_id,
    dat.ped_mean_g12 mean_x12,
    dat.ped_rms_g12 rms_x12,
    dat.ped_mean_g6 mean_x6,
    dat.ped_rms_g6 rms_x6,
    dat.ped_mean_g1 mean_x1,
    dat.ped_rms_g1 rms_x1
  FROM 
    /* inline view selects the mon_run_iov with the greatest subrun_num */
    (SELECT iov_id, run_iov_id, 
       MAX(subrun_num) KEEP (DENSE_RANK FIRST ORDER BY subrun_num ASC)
       FROM mon_run_iov@cmsomds GROUP BY iov_id, run_iov_id) miov
    JOIN run_iov@cmsomds riov ON miov.run_iov_id = riov.iov_id
    JOIN run_tag@cmsomds rtag ON riov.tag_id = rtag.tag_id
    JOIN run_type_def@cmsomds rdef ON rtag.run_type_id = rdef.def_id
    JOIN location_def@cmsomds ldef ON rtag.location_id = ldef.def_id
    JOIN mon_pedestals_dat@cmsomds dat ON dat.iov_id = miov.iov_id
    JOIN channelview@cmsomds cv ON cv.logic_id = dat.logic_id
  WHERE 
        cv.name='Offline_det_id'
    AND ldef.location='H4'
    AND rdef.run_type='PEDESTAL'
    AND rdef.config_tag='PEDESTAL-STD'
    AND rdef.config_ver=1
;
