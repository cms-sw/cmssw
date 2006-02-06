/** \file
 * Implementation of class RPCRecordFormatter
 *
 *  $Date: 2006/02/06 09:25:19 $
 *  $Revision: 1.10 $
 *
 * \author Ilaria Segoni
 */

#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecord.h"
#include "EventFilter/RPCRawToDigi/interface/RPCBXData.h"
#include "EventFilter/RPCRawToDigi/interface/RPCChamberData.h"
#include "EventFilter/RPCRawToDigi/interface/RPCChannelData.h"
#include "EventFilter/RPCRawToDigi/interface/RMBErrorData.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include <vector>

RPCRecordFormatter::RPCRecordFormatter(bool printout)
:currentBX(0){
	verbosity=printout;
}

RPCRecordFormatter::~RPCRecordFormatter(){
}

void RPCRecordFormatter::recordUnpack(RPCRecord::recordTypes typeOfRecord, const unsigned char* recordIndex, 
std::auto_ptr<RPCDigiCollection> prod){

   int bx=0;
   RPCDetId detId;

   if(typeOfRecord==RPCRecord::StartOfBXData)      {
	currentBX = this->unpackBXRecord(recordIndex);
   }	   
    
    if(typeOfRecord==RPCRecord::StartOfChannelData) {
	currentDetId =this->unpackChannelRecord(recordIndex);
    }
   
    /// Unpacking Strips With Hit
    if(typeOfRecord==RPCRecord::ChamberData)	    {
	std::vector<int> stripsOn=this->unpackChamberRecord(recordIndex);
	for(std::vector<int>::iterator pStrip = stripsOn.begin(); pStrip !=
    		      stripsOn.end(); ++pStrip){

		int strip = *(pStrip);
		/// Creating RPC digi
		RPCDigi digi(strip,currentBX);

		/// Committing to the product
		prod->insertDigi(currentDetId,digi);
          }
    }
    
    if(typeOfRecord==RPCRecord::RMBDiscarded) this->unpackRMBCorruptedRecord(recordIndex);
    if(typeOfRecord==RPCRecord::DCCDiscarded) rpcData.addDCCDiscarded();
    }




int RPCRecordFormatter::unpackBXRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

    RPCBXData bxData(recordIndexInt);
    if(verbosity) std::cout<<"Found BX record, BX= "<<bxData.bx()<<std::endl;
    return bxData.bx();
    rpcData.addBXData(bxData);

} 


RPCDetId RPCRecordFormatter::unpackChannelRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

    RPCChannelData chnData(recordIndexInt);
    if(verbosity) std::cout<<"Found start of Channel Data Record, Channel: "<< chnData.channel()<<
 	 " Readout/Trigger Mother Board: "<<chnData.tbRmb()<<std::endl;
    
    RPCDetId detId/*=chnData.detId()*/;
    return detId;
    
    rpcData.addChnData(chnData);

} 

std::vector<int> RPCRecordFormatter::unpackChamberRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

   RPCChamberData cmbData(recordIndexInt);
    if(verbosity) std::cout<< "Found Chamber Data, Chamber Number: "<<cmbData.chamberNumber()<<
 	" Partition Data "<<cmbData.partitionData()<<
 	" Half Partition " << cmbData.halfP()<<
 	" Data Truncated: "<<cmbData.eod()<<
 	" Partition Number " <<  cmbData.partitionNumber()
 	<<std::endl;
	
    vector<int> stripID/*=cmbData.getStrips()*/;
    return stripID;
    
    rpcData.addRPCChamberData(cmbData);
}



void RPCRecordFormatter::unpackRMBCorruptedRecord(const unsigned char* recordIndex) {



const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

     RMBErrorData  discarded(recordIndexInt);
     rpcData.addRMBDiscarded(discarded);
     rpcData.addRMBCorrupted(discarded);

 }



