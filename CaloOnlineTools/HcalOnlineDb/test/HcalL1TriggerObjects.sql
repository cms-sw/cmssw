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
      AVERAGE_PEDESTAL, 
      RESPONSE_CORRECTED_GAIN, 
      FLAG, 
      'fake_metadata_name', 
      'fake_metadata_value' 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
            MAX(theview.interval_of_validity_begin) as iov_begin, 
            theview.channel_map_id 
     from 
            cms_hcl_hcal_cond.v_hcal_L1_TRIGGER_OBJECTS theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
     group by 
            theview.channel_map_id 
     order by 
            theview.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS sp 
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
       -999999.0, 
       -999999.0, 
       -999999, 
       TRIGGER_OBJECT_METADATA_NAME,
       TRIGGER_OBJECT_METADATA_VALUE 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
	    MAX(theview.interval_of_validity_begin) as iov_begin 
     from 
            cms_hcl_hcal_cond.v_hcal_L1_TRIGGER_OBJECTS_MDA theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS_MDA sp 
on 
fp.record_id=sp.record_id 
 
