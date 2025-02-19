        select 
               sp.objectname 
               ,sp.subdet as subdetector 
               ,sp.ieta as IETA 
               ,sp.iphi as IPHI 
               ,sp.depth as DEPTH 
               ,sp.type 
               ,sp.section 
               ,sp.ispositiveeta 
               ,sp.sector 
               ,sp.module 
               ,sp.channel_on_off_state as ON_OFF 
               ,sp.channel_status_word as STATUS_WORD 
    from 
               ( 
               select 
                      MAX(cq.record_id) as record_id 
                      ,MAX(cq.interval_of_validity_begin) as iov_begin 
                      ,cq.channel_map_id 
               from 
                      cms_hcl_hcal_cond.v_hcal_channel_quality cq 
               where 
                      tag_name=:1 
               --and 
                      --cq.VERSION=:2 
               AND 
                      cq.interval_of_validity_begin<=:2 
               group by 
                      cq.channel_map_id 
               order by 
                      cq.channel_map_id 
               ) fp 
        inner join 
               cms_hcl_hcal_cond.v_hcal_channel_quality sp 
        on 
               fp.record_id=sp.record_id 
