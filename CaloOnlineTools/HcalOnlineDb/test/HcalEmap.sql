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
      CHANNEL as i, 
      cr, 
      sl, 
      tb, 
      dcc, 
      spigot, 
      fiber, 
      fiberchan 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
            MAX(theview.interval_of_validity_begin) as iov_begin, 
            theview.channel_map_id 
     from 
            cms_hcl_hcal_cond.V_HCAL_EMAP_V1 theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
     group by 
            theview.channel_map_id 
     order by 
            theview.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_EMAP_V1 sp 
on 
fp.record_id=sp.record_id 
 
