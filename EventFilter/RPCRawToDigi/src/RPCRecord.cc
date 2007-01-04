/** \file
 *
 *  Implementation of RPCRecord Class
 *
 *  $Date: 2006/10/08 12:11:29 $
 *  $Revision: 1.9 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>


RPCRecord::recordTypes RPCRecord::computeType(){ 
    
wordType = UndefinedType;
    
/// Link Board Data
//LogDebug("recordType intiial:") << " record type:  " << wordType; 
if ( (int)((*word_ >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) <= MaxLBFlag) wordType = LinkBoardData;
 
//LogDebug("recordType intiial:") << " record type:  " << wordType; 
/// Control Word
if ( (int)((*word_ >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) == controlWordFlag){
      
//LogDebug("recordTypecheck, is control :") << " record type:  " << wordType; 
	/// StartOfBXData
	if ( (int)((*word_ >> BX_TYPE_SHIFT) & BX_TYPE_MASK) == BXFlag) wordType = StartOfBXData;
//LogDebug("recordType check bx:") << " record type:  " << wordType; 
	/// StartOfTbLinkInputNumberData             
	if ( (int)((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == StartOfLBInputDataFlag) wordType =  StartOfTbLinkInputNumberData;
	///  RMBDiscarded           
	if ( (int) ((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == RMBDiscardedDataFlag  ) wordType = RMBDiscarded;
	///  RMBCorrupted           
	if ( (int) ((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == RMBCorruptedDataFlag  ) wordType = RMBCorrupted;
	/// Empty or DCC Discarded 
	if ( (int)((*word_ >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == EmptyOrDCCDiscardedFlag){
       
		if ( (int)((*word_ >>  EMPTY_OR_DCCDISCARDED_SHIFT) & EMPTY_OR_DCCDISCARDED_MASK) == EmptyWordFlag) wordType = EmptyWord;
		if ( (int) ((*word_ >> EMPTY_OR_DCCDISCARDED_SHIFT) & EMPTY_OR_DCCDISCARDED_MASK) == DCCDiscardedFlag) wordType = DCCDiscarded;
		if ( (int) ((*word_ >> RMB_DISABLED_SHIFT) & RMB_DISABLED_MASK) == RMBDisabledDataFlag) wordType = RMBDisabled;
	}
}

//LogDebug("recordType final:") << " record type:  " << wordType; 
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
