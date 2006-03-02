/* query to fill ECALPEDESTALS table */

  SELECT 
    miov.iov_id iov_value_id, 
    riov.run_num time
  FROM 
    location_def ldef, run_type_def rdef, run_tag rtag, run_iov riov,
    /* inline view selects the mon_run_iov with the greatest subrun_num */
    (SELECT iov_id, run_iov_id, 
       MAX(subrun_num) KEEP (DENSE_RANK FIRST ORDER BY subrun_num ASC)
       FROM mon_run_iov GROUP BY iov_id, run_iov_id) miov
  WHERE
      miov.run_iov_id = riov.iov_id
  AND riov.tag_id = rtag.tag_id
  AND rdef.def_id = rtag.run_type_id
  AND ldef.def_id = rtag.location_id
  AND ldef.location='H4'
  AND rdef.run_type='PEDESTAL'
  AND rdef.config_tag='PEDESTAL-STD'
  AND rdef.config_ver=1
;

/* query to fill ECALPEDESTALS_ITEM table */
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
    location_def ldef, run_type_def rdef, run_tag rtag, run_iov riov,
    /* inline view selects the mon_run_iov with the greatest subrun_num */
    (SELECT iov_id, run_iov_id, 
       MAX(subrun_num) KEEP (DENSE_RANK FIRST ORDER BY subrun_num ASC)
       FROM mon_run_iov GROUP BY iov_id, run_iov_id) miov,
    mon_pedestals_dat dat, channelview cv
  WHERE dat.iov_id = miov.iov_id
    AND cv.logic_id = dat.logic_id
    AND cv.name='Offline_det_id'
    AND miov.run_iov_id = riov.iov_id
    AND riov.tag_id = rtag.tag_id
    AND rdef.def_id = rtag.run_type_id
    AND ldef.def_id = rtag.location_id
    AND ldef.location='H4'
    AND rdef.run_type='PEDESTAL'
    AND rdef.config_tag='PEDESTAL-STD'
    AND rdef.config_ver=1
;
