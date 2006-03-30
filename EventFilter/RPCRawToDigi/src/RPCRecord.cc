/** \file
 *
 *  Implementation of RPCRecord Class
 *
 *  $Date: 2005/12/15 17:49:32 $
 *  $Revision: 1.6 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h>

#include <iostream>


RPCRecord::recordTypes RPCRecord::computeType(){ 
    
wordType = UndefinedType;
    
/// Link Board Data
if ( (int)((*word_ >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) <= MaxLBFlag) wordType = LinkBoardData;
 
/// Control Word
if ( (int)((*word_ >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) == controlWordFlag){
      
	/// StartOfBXData
	if ( (int)((*word_ >> BX_TYPE_SHIFT) & BX_TYPE_MASK) == BXFlag) wordType = StartOfBXData;
	/// StartOfChannelData             
	if ( (int)((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == StartOfChannelDataFlag) wordType = StartOfChannelData;
	///  RMBDiscarded           
	if ( (int) ((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == RMBDiscardedDataFlag  ) wordType = RMBDiscarded;
	///  RMBCorrupted           
	if ( (int) ((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == RMBCorruptedDataFlag  ) wordType = RMBCorrupted;
	/// Empty or DCC Discarded 
	if ( (int)((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == EmptyOrDCCDiscardedFlag){
       
		if ( (int)((*word_ >> EMPTY_OR_DCCDISCARDED_SHIFT) & EMPTY_OR_DCCDISCARDED_MASK) == EmptyWordFlag) wordType = EmptyWord;
		if ( (int) ((*word_ >> EMPTY_OR_DCCDISCARDED_SHIFT) & EMPTY_OR_DCCDISCARDED_MASK) == DCCDiscardedFlag) wordType = DCCDiscarded;
	}
}

return wordType;
}


bool RPCRecord::check(){

 if((oldRecord<3) & (wordType != oldRecord+1)) return true;
 return false;

}

RPCRecord::recordTypes RPCRecord::type(){ 
return wordType;
}

const unsigned int * RPCRecord::buf(){
return word_;
} 
