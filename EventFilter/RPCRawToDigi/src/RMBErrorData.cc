/** \file
 *
 *  $Date: 2005/11/07 15:42:14 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RMBErrorData.h>

RMBErrorData::RMBErrorData(const unsigned char* index): 
    word_(reinterpret_cast<const unsigned int*>(index)) {
    
    channel_= (*word_>> CHANNEL_SHIFT )& CHANNEL_MASK;
    tbRmb_  = (*word_>> TB_RMB_SHIFT)  & TB_RMB_MASK;
    chamber_= (*word_>> CHAMBER_SHIFT)& CHAMBER_MASK;

}


int RMBErrorData::channel(){  
  return channel_;
}

int RMBErrorData::tbRmb(){  
  return tbRmb_;
}

int RMBErrorData::chamber(){  
  return chamber_;
}

