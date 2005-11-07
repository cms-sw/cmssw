/** \file
 *
 *  $Date: 2005/10/21 17:14:32 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/StartOfChannelData.h>

StartOfChannelData::StartOfChannelData(const unsigned char* index): 
    word_(reinterpret_cast<const unsigned int*>(index)) {
    
    channel_= (*word_>> CHANNEL_SHIFT )& CHANNEL_MASK;
    tbRmb_ =  (*word_>> TB_RMB_SHIFT)  & TB_RMB_MASK;
    chamber_= (*word_>> CHAMBER_SHIFT)& CHAMBER_MASK;

}


int StartOfChannelData::channel(){  
  return channel_;
}

int StartOfChannelData::tbRmb(){  
  return tbRmb_;
}

int StartOfChannelData::chamber(){  
  return chamber_;
}

