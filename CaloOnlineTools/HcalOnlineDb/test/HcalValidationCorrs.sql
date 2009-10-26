SELECT 
      OBJECTNAME, 
      SUBDET, 
      IETA, 
      IPHI, 
      DEPTH, 
      TYPE, 
      SECTION, 
      ISPOSITIVEETA, 
      SECTOR, 
      MODULE, 
      CHANNEL, 
      VALUE 
FROM ( 
     select 
            MIN(rc.record_id) as record_id, 
	    MAX(rc.interval_of_validity_begin) as iov_begin, 
	    rc.channel_map_id 
     from 
            cms_hcl_hcal_cond.v_hcal_validation_corrections rc 
     where 
            tag_name=:1 
     AND 
            rc.interval_of_validity_begin<=:2 
     group by 
            rc.channel_map_id 
     order by 
            rc.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_VALIDATION_CORRECTIONS sp 
on 
fp.record_id=sp.record_id 
