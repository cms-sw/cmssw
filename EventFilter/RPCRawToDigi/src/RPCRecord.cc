/** \file
 *
 *  Implementation of RPCRecord Class
 *
 *  $Date: 2005/11/09 11:35:09 $
 *  $Revision: 1.4 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/RPCBXData.h>
#include <EventFilter/RPCRawToDigi/interface/RMBErrorData.h>
#include <EventFilter/RPCRawToDigi/interface/RPCChannelData.h>
#include <EventFilter/RPCRawToDigi/interface/RPCChamberData.h>

#include <iostream>

using namespace std;


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


void RPCRecord::recordUnpack(recordTypes  type){

/// BX Data type
 if(type==StartOfBXData){
    RPCBXData bxData(word_);
    if(verbosity) cout<<"Found BX record, BX= "<<bxData.bx()<<endl;

   // rpcData.addBXData(bxData);
 } 

/// Start of Channel Data Type
 if(type==StartOfChannelData){
    RPCChannelData chnData(word_);
    if(verbosity) cout<<"Found start of Channel Data Record, Channel: "<< chnData.channel()<<
 	 " Readout/Trigger Mother Board: "<<chnData.tbRmb()<<endl;
 } 

/// Chamber Data 
 if(type==ChamberData){
   RPCChamberData cmbData(word_);
    if(verbosity) cout<< "Found Chamber Data, Chamber Number: "<<cmbData.chamberNumber()<<
 	" Partition Data "<<cmbData.partitionData()<<
 	" Half Partition " << cmbData.halfP()<<
 	" Data Truncated: "<<cmbData.eod()<<
 	" Partition Number " <<  cmbData.partitionNumber()
 	<<endl;
 }

/// RMB Discarded
 if(type==RMBDiscarded){
  RMBErrorData  discarded(word_);
  //	rpcData.addRMBDiscarded(discarded);
  //   rpcData.addRMBCorrupted(corrupted);
 }

/// DCC Discraded
 if(type==DCCDiscarded){
    // rpcData.addDCCDiscarded();
 }

}
