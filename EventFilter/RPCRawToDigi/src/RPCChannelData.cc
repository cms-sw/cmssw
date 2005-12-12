/** \file
 *
 * Implementation of RPCChannelData
 *
 *  $Date: 2005/11/24 18:17:50 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCChannelData.h>

RPCChannelData::RPCChannelData(const unsigned int* index): 
    word_(index) {
    
    channel_= (*word_>> CHANNEL_SHIFT )& CHANNEL_MASK;
    tbRmb_ =  (*word_>> TB_RMB_SHIFT)  & TB_RMB_MASK;

 }


int RPCChannelData::channel(){  
  return channel_;
}

int RPCChannelData::tbRmb(){  
  return tbRmb_;
}

