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
      rec_hit_calibration, 
      lut_granularity, 
      output_lut_threshold 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
            MAX(theview.interval_of_validity_begin) as iov_begin, 
            theview.channel_map_id 
     from 
            cms_hcl_hcal_cond.v_hcal_lut_chan_data_v1 theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
     group by 
            theview.channel_map_id 
     order by 
            theview.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_lut_chan_data_v1 sp 
on 
fp.record_id=sp.record_id 
 
union 
 
SELECT 
       'fakeobjectname', 
       'fakesubdetector', 
       -1, 
       -1, 
       -1, 
       -1, 
       'fakesection', 
       -1, 
       -1, 
       -1, 
       -1, 
       rctlsb, 
       nominal_gain, 
       -1 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
	    MAX(theview.interval_of_validity_begin) as iov_begin 
     from 
            cms_hcl_hcal_cond.v_hcal_lut_metadata_v1 theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_lut_metadata_v1 sp 
on 
fp.record_id=sp.record_id 
 
