/** \file
 *
 *  $Date: 2005/11/09 11:36:21 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCBXData.h>
#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>


RPCBXData::RPCBXData(const unsigned char* index): 
    word_(reinterpret_cast<const unsigned int*>(index)) {
    
    bx_= (*word_ >> BX_SHIFT )& BX_MASK ;

//    index++RPC_RECORD_BIT_SIZE;
    while(1){
   
     RPCRecord record(index);
     RPCRecord::recordTypes typeOfRecord = record.type();     
     if(typeOfRecord==RPCRecord::StartOfChannelData)
     {
	
	RPCChannelData unpackedChannelData(index);
	channelsData.push_back(unpackedChannelData);
     
     }else{
     	
	//logFile<<"No data for BX "<<bx_<<endl;
     	break;
     
     }


     index += RPCRecord::RPC_RECORD_BIT_SIZE;
   
   
   }
}


int RPCBXData::bx(){  
  return bx_;
}

