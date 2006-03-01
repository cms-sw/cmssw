


CREATE OR REPLACE PROCEDURE payload_o2o AS

obj  VARCHAR2(100);
id   NUMBER;
cnt1 NUMBER;
cnt2 NUMBER;

BEGIN
  -- EcalPedestals
  obj  := 'EcalPedestals';
  SELECT count(*) INTO cnt1 FROM cms_streamuser.ecalpedestals;
  SELECT max(last_id) INTO id FROM o2o_log WHERE object_name = obj;

  IF id IS NULL THEN
    -- it is the first transfer, ensure ALL ids are transferred
    -- use -1 because all iov_id are > 0
    id := -1;
  END IF;

  -- execute transfer
  INSERT INTO cms_streamuser.ecalpedestals
  (iov_value_id, time)
  SELECT 
    iov.iov_id, iov.run_num
  FROM 
    location_tag ltag, run_tag rtag, run_iov iov,
    (SELECT DISTINCT iov_id FROM mon_pedestals_dat) dat
  WHERE ltag.tag_id = rtag.location_id 
    AND iov.tag_id = rtag.tag_id
    AND dat.iov_id = iov.iov_id
    AND ltag.location='H4' 
    AND rtag.run_type='PEDESTAL'
    AND iov.iov_id > id
;
    
  INSERT INTO cms_streamuser.ecalpedestals_value
  (iov_value_id, pos, det_id, mean_x12, rms_x12, mean_x6, rms_x6, mean_x1, rms_x1)
  SELECT 
    dat.iov_id,
    ROW_NUMBER() OVER (PARTITION BY dat.iov_id ORDER BY dat.logic_id ASC),
    cv.id1,
    dat.ped_mean_g12,
    dat.ped_rms_g12,
    dat.ped_mean_g6,
    dat.ped_rms_g6,
    dat.ped_mean_g1,
    dat.ped_rms_g1
  FROM 
    cms_streamuser.ecalpedestals iov, mon_pedestals_dat dat, channelview cv
  WHERE dat.iov_id = iov.iov_value_id
    AND cv.logic_id = dat.logic_id
    AND cv.name='Offline_det_id'
    AND iov.iov_value_id > id
;

  -- update log
  SELECT count(*) INTO cnt2 FROM cms_streamuser.ecalpedestals;
  IF cnt2 != cnt1 THEN
    SELECT max(iov_value_id) INTO id FROM cms_streamuser.ecalpedestals;
    INSERT INTO o2o_log (object_name, last_id, num_transfered, transfer_time)
      VALUES (obj, id, cnt2 - cnt1, sysdate);
  END IF;

  COMMIT;
END;
/

SHOW ERRORS;
