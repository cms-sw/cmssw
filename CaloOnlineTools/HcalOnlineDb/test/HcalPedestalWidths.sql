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
      IS_ADC_COUNTS, 
      COVARIANCE_00, 
      COVARIANCE_01, 
      COVARIANCE_02, 
      COVARIANCE_03, 
      COVARIANCE_10, 
      COVARIANCE_11, 
      COVARIANCE_12, 
      COVARIANCE_13, 
      COVARIANCE_20, 
      COVARIANCE_21, 
      COVARIANCE_22, 
      COVARIANCE_23, 
      COVARIANCE_30, 
      COVARIANCE_31, 
      COVARIANCE_32, 
      COVARIANCE_33 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
            MAX(theview.interval_of_validity_begin) as iov_begin, 
            theview.channel_map_id 
     from 
            cms_hcl_hcal_cond.V_HCAL_PEDESTAL_WIDTHS_V3 theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
     group by 
            theview.channel_map_id 
     order by 
            theview.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_PEDESTAL_WIDTHS_V3 sp 
on 
fp.record_id=sp.record_id 
 
