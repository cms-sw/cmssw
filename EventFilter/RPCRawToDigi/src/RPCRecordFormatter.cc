/** \file
 * Implementation of class RPCRecordFormatter
 *
 *  $Date: 2006/02/15 09:41:14 $
 *  $Revision: 1.4 $
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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

RPCRecordFormatter::RPCRecordFormatter(bool printout){
	verbosity=printout;
	currentBx=0;
}

RPCRecordFormatter::~RPCRecordFormatter(){
}

/// Note that it takes a reference to std::auto_ptr<RPCDigiCollection> because
/// I don't want to transfer ownership of RPCDigiCollection (I.S.)
void RPCRecordFormatter::recordUnpack(RPCRecord::recordTypes typeOfRecord, const unsigned char* recordIndex, 
std::auto_ptr<RPCDigiCollection> & prod){
    
       
   if(typeOfRecord==RPCRecord::StartOfBXData)      {
	currentBx = this->unpackBXRecord(recordIndex);
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
		RPCDigi digi(strip, currentBx);

		/// Committing digi to the product
		prod->insertDigi(currentDetId,digi);
          }
    }
    
    if(typeOfRecord==RPCRecord::RMBDiscarded) this->unpackRMBCorruptedRecord(recordIndex);
    if(typeOfRecord==RPCRecord::DCCDiscarded) rpcData.addDCCDiscarded();
    }




int RPCRecordFormatter::unpackBXRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

    RPCBXData bxData(recordIndexInt);
    edm::LogInfo ("RPCUnpacker")<<"Found BX record, BX= "<<bxData.bx();
    return bxData.bx();
    rpcData.addBXData(bxData);

} 


RPCDetId RPCRecordFormatter::unpackChannelRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

    RPCChannelData chnData(recordIndexInt);
    edm::LogInfo ("RPCUnpacker")<<"Found start of Channel Data Record, Channel: "<< chnData.channel()<<
 	 " Readout/Trigger Mother Board: "<<chnData.tbRmb();
    
    rpcData.addChnData(chnData);

    ///Temporary Phony RPCDetId
    int region=0, ring=-1, station=1, sector=1, layer =1, subsector =1, roll=2;
    RPCDetId detId(region, ring, station, sector, layer, subsector, roll);
    //RPCDetId detId/*=chnData.detId()*/;
    return detId;



} 

std::vector<int> RPCRecordFormatter::unpackChamberRecord(const unsigned char* recordIndex) {

const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

   RPCChamberData cmbData(recordIndexInt);
    edm::LogInfo ("RPCUnpacker")<< "Found Chamber Data, Chamber Number: "<<cmbData.chamberNumber()<<
 	" Partition Data "   <<cmbData.partitionData()<<
 	" Half Partition "   << cmbData.halfP()<<
 	" Data Truncated: "  << cmbData.eod()<<
 	" Partition Number " <<  cmbData.partitionNumber();
    
    rpcData.addRPCChamberData(cmbData);

    vector<int> stripID/*=cmbData.getStrips()*/;
    ///Strip Id
    int partData= cmbData.partitionData();
    int partNumb= cmbData.partitionNumber();
    for (int ii=0; ii<8; ii++) {
    	if ((partData>>ii) & 0x1) stripID.push_back(partNumb*8 + ii);
    }
    

    return stripID;
}



void RPCRecordFormatter::unpackRMBCorruptedRecord(const unsigned char* recordIndex) {



const unsigned int* recordIndexInt=reinterpret_cast<const unsigned int*>(recordIndex);

     RMBErrorData  discarded(recordIndexInt);
     rpcData.addRMBDiscarded(discarded);
     rpcData.addRMBCorrupted(discarded);

 }



