CREATE OR REPLACE PROCEDURE payload_o2o AS

obj  VARCHAR2(100);
id   NUMBER;
cnt1 NUMBER;
cnt2 NUMBER;
start_time TIMESTAMP;

BEGIN
  -- EcalPedestals
  obj  := 'EcalPedestals';
  SELECT count(*) INTO cnt1 FROM ecalpedestals@devdb10.cern.ch;
  SELECT max(last_id) INTO id FROM o2o_log WHERE object_name = obj;

  IF id IS NULL THEN
    -- it is the first transfer, ensure ALL ids are transferred
    -- use -1 because all iov_id are > 0
    id := -1;
  END IF;

  -- get the start time
  SELECT systimestamp INTO start_time FROM dual;

  -- execute transfer
  INSERT INTO ecalpedestals@devdb10.cern.ch
  (iov_value_id, time)
  SELECT 
    miov.iov_id, riov.run_num
  FROM 
    location_def ldef, run_type_def rdef, run_tag rtag, run_iov riov,
    /* selects the mon_run_iov with the greatest subrun_num */
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
  AND miov.iov_id > id
;
    
  INSERT INTO ecalpedestals_item@devdb10.cern.ch
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
    ecalpedestals@devdb10.cern.ch iov, mon_pedestals_dat dat, channelview cv
  WHERE dat.iov_id = iov.iov_value_id
    AND cv.logic_id = dat.logic_id
    AND cv.name='Offline_det_id'
    AND iov.iov_value_id > id
;

  -- update log
  SELECT count(*) INTO cnt2 FROM ecalpedestals@devdb10.cern.ch;
  IF cnt2 != cnt1 THEN
    SELECT max(iov_value_id) INTO id FROM ecalpedestals@devdb10.cern.ch;
    INSERT INTO o2o_log (object_name, last_id, num_transfered, transfer_start, transfer_duration)
      VALUES (obj, id, cnt2 - cnt1, start_time, start_time - systimestamp);
  END IF;

  COMMIT;
END;
/

SHOW ERRORS;
