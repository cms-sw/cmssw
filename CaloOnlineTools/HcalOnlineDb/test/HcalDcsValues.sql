select 
      i.dpname, 
      -1 as lumisection, 
      i.value, 
      i.set_high as upper, 
      i.set_low as lower, 
      REGEXP_SUBSTR(i.DPNAME, 'H[BEFO]', 1, 1, 'c') as subdetector, 
      i.ring, 
      TO_NUMBER(LTRIM(i.slice, 'SQ')) as slice, 
      TO_NUMBER(LTRIM(i.subchannel, 'PRM')) as subchannel, 
      i.type 
from 
      cms_hcl_hcal_cond.V_CMS_HCAL_HV_INIT_VALUES i 
where 
      i.tag_name=:1 
      and 'fakeversion' like :2 
      and 1 = :3 
      and i.run_number=:4 
UNION 
select 
      i.dpname, 
      i.lum_secxn as lumisection, 
      i.value, 
      i.set_high as upper, 
      i.set_low as lower, 
      REGEXP_SUBSTR(i.DPNAME, 'H[BEFO]', 1, 1, 'c') as subdetector, 
      i.ring, 
      TO_NUMBER(LTRIM(i.slice, 'SQ')) as slice, 
      TO_NUMBER(LTRIM(i.subchannel, 'PRM')) as subchannel, 
      i.type 
from 
      cms_hcl_hcal_cond.V_CMS_HCAL_HV_UPDATE_VALUES i 
where 
      i.tag_name=:1 
      and 'fakeversion' like :2 
      and 1 = :3 
      and i.run_number=:4 
