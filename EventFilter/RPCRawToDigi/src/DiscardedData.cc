/** \file
 *
 *  $Date: 2005/10/21 17:14:32 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/DiscardedData.h>

DiscardedData::DiscardedData(const unsigned char* index): 
    word_(reinterpret_cast<const unsigned int*>(index)) {
    
    channelDiscarded_= (*word_>> CHANNEL_DISCARDED_SHIFT )& CHANNEL_DISCARDED_MASK;
    tbRmbDiscarded_ =  (*word_>> TB_RMB_DISCARDED_SHIFT)  & TB_RMB_DISCARDED_MASK;
    chamberDiscarded_= (*word_>> CHAMBER_DISCARDED_SHIFT)& CHAMBER_DISCARDED_MASK;

}


int DiscardedData::channelDiscarded(){  
  return channelDiscarded_;
}

int DiscardedData::tbRmbDiscarded(){  
  return tbRmbDiscarded_;
}

int DiscardedData::chamberDiscarded(){  
  return chamberDiscarded_;
}

