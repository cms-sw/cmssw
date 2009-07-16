// -*- C++ -*-
//
// Package:     HcalTPGAlgos
// Class  :     HcalChannelQualityManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Fri Jul 10 10:39:06 CEST 2009
// $Id: HcalChannelQualityManager.cc,v 1.1 2009/07/10 10:24:12 kukartse Exp $
//

#include "CalibCalorimetry/HcalTPGAlgos/interface/HcalChannelQualityManager.h"

HcalChannelQualityManager::HcalChannelQualityManager()
{
}

// HcalChannelQualityManager::HcalChannelQualityManager(const HcalChannelQualityManager& rhs)
// {
//    // do actual copying here;
// }

HcalChannelQualityManager::~HcalChannelQualityManager()
{
}

bool HcalChannelQualityManager::isChannelMasked(DetId channel, bool testmode){
  bool isMasked = false;
  //
  //_____ true is returned for a few channels for testing
  //
  if (testmode){
    //detid subdet ieta iphi depth
    //13401380 HB -1 1 1
    // trigger: ieta iphi
    //13408717 -32 1
    if (channel.rawId() == 13401380 || channel.rawId() == 13408717){
      isMasked = true;
    }
  }
  //
  //_____ normal mode of operation: true for every channel masked in ChannelQuality
  //
  else{
  }
  return isMasked;
}


/* SQL query to extract channel on/off status

select 
       sp.record_id as record_id
       ,sp.channel_map_id detid
       ,sp.subdet as subdetector
       ,sp.ieta as IETA
       ,sp.iphi as IPHI
       ,sp.depth as DEPTH
       ,sp.interval_of_validity_begin as IOV_BEGIN
       ,sp.interval_of_validity_end as IOV_END
       ,sp.channel_on_off_state as ON_OFF
       ,sp.channel_status_word as STATUS_WORD
       ,sp.commentdescription 
from
(
select MAX(cq.record_id) as record_id
       ,MAX(cq.interval_of_validity_begin) as iov_begin
       ,cq.channel_map_id
from
       cms_hcl_hcal_cond.v_hcal_channel_quality cq
where
       tag_name='AllChannelsMasked16Jul2009v1'
group by 
       cq.channel_map_id
order by
       cq.channel_map_id
) fp
inner join
       cms_hcl_hcal_cond.v_hcal_channel_quality sp
on
       fp.record_id=sp.record_id

*/
