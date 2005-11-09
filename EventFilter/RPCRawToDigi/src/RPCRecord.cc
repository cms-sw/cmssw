/** \file
 *
 *  $Date: 2005/11/03 15:23:54 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>


RPCRecord::recordTypes RPCRecord::type(){ 
    
enum recordTypes wordType = UndefinedType;
    
/// Chamber Data
if ( (int)((*word_ >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) <= MaxChamberFlag) wordType = ChamberData;
 
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


void RPCRecord::next() 
{ 
	word_ += RPC_RECORD_BIT_SIZE; 
}

