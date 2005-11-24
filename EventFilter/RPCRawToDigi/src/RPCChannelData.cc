/** \file
 *
 *  $Date: 2005/11/07 15:42:32 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCChannelData.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>

RPCChannelData::RPCChannelData(const unsigned char* index): 
    word_(reinterpret_cast<const unsigned int*>(index)) {
    
    channel_= (*word_>> CHANNEL_SHIFT )& CHANNEL_MASK;
    tbRmb_ =  (*word_>> TB_RMB_SHIFT)  & TB_RMB_MASK;
    chamber_= (*word_>> CHAMBER_SHIFT)& CHAMBER_MASK;

    while(1){
   
     RPCRecord record(index);
     RPCRecord::recordTypes typeOfRecord = record.type();     
     if(typeOfRecord==RPCRecord::ChamberData)
     {
	
	ChamberData unpackedChamberData(index);
	chambersData.push_back(unpackedChamberData);
     
     }else{
     	
	//logFile<<"CD following SCD!! "<<endl;
	//must be treated as error;
     	break;
     
     }


     record.next();
   
   }
}


int RPCChannelData::channel(){  
  return channel_;
}

int RPCChannelData::tbRmb(){  
  return tbRmb_;
}

int RPCChannelData::chamber(){  
  return chamber_;
}

