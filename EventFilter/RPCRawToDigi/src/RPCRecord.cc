/** \file
 *
 *  $Date: 2005/10/21 17:14:32 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>

//RPCRecord::RPCRecord(const unsigned char* index){

//    word_(reinterpret_cast<const unsigned int*>(index));

//}



RPCRecord::recordTypes RPCRecord::type(){ 
    
    enum recordTypes wordType = UndefinedType;
    
    // Chamber Data
      if ( ((*word_ >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) <= MaxChamberFlag) wordType = ChamberData;
 
    // Control Word
      if ( ((*word_ >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) == controlWordFlag){
      
              
       if ( ((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == StartOfChannelDataFlag) wordType = StartOfChannelData;
       if ( ((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == RMBDiscardedDataFlag)   wordType = RMBDiscardedDataFlag;
       
       //Discarded Data
       if ( ((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == EmptyWordOrSLinkDiscardedFlag){
       
        if ( ((*word_ >> EMPTY_OR_SLDISCARDED_SHIFT) & EMPTY_OR_SLDISCARDED_MASK) == EmptyWordFlag) wordType = EmptyWord;
        if ( ((*word_ >> EMPTY_OR_SLDISCARDED_SHIFT) & EMPTY_OR_SLDISCARDED_MASK) == SLinkDiscardedDataFlag) wordType = SLinkDiscardedData;
      
       
       
       }
    
      
      
      }

    return wordType;
}


void RPCRecord::next() 
{ 
	word_ += RPC_RECORD_BIT_SIZE; 
}

