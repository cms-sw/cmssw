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
#include "EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h"
#include "EventFilter/RPCRawToDigi/interface/RMBErrorData.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRawDataPattern.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

RPCRecordFormatter::RPCRecordFormatter():currentRMB(0),currentChannel(0){
}

RPCRecordFormatter::~RPCRecordFormatter(){
}

/// Note that it takes a reference to std::auto_ptr<RPCDigiCollection> because
/// I don't want to transfer ownership of RPCDigiCollection (I.S.)
void RPCRecordFormatter::recordUnpack(RPCRecord & theRecord,
		std::auto_ptr<RPCDigiCollection> & prod, RPCFEDData & rawData){
    
   int bx=0;
   ///Temporary Phony RPCDetId
   int region=0, ring=-1, station=1, sector=1, layer =1, subsector =1, roll=2;
   RPCDetId DetId(region, ring, station, sector, layer, subsector, roll);

   enum RPCRecord::recordTypes typeOfRecord = theRecord.type();
   const unsigned int* recordIndexInt= theRecord.buf();

   if(typeOfRecord==RPCRecord::StartOfBXData)      {
	bx = this->unpackBXRecord(recordIndexInt);
        rawData.addBXData(bx);
   }	   
    
    if(typeOfRecord==RPCRecord::StartOfChannelData) {
    	currentRMB=0;
	currentChannel=0;
	this->unpackChannelRecord(recordIndexInt);
    }
   
    /// Unpacking BITS With Hit (uniquely related to strips with hit)
    if(typeOfRecord==RPCRecord::LinkBoardData)	    {
	RPCLinkBoardData lbData=this->unpackLBRecord(recordIndexInt);
    	rawData.addRMBData(currentRMB,currentChannel, lbData);  

	std::vector<int> bits=lbData.bitsOn();
	for(std::vector<int>::iterator pBit = bits.begin(); pBit !=
    		      bits.end(); ++pBit){

		int bit = *(pBit);
		/// Creating RPC digi
		/// When channel mapping available calculate strip
		///and replace bit with strip
		RPCDigi digi(bit,bx);

		/// Committing digi to the product
		prod->insertDigi(DetId,digi);
          }
    }
    
    if(typeOfRecord==RPCRecord::RMBDiscarded || typeOfRecord==RPCRecord::RMBCorrupted) this->unpackRMBCorruptedRecord(recordIndexInt,typeOfRecord,rawData);
    if(typeOfRecord==RPCRecord::DCCDiscarded) rawData.addDCCDiscarded();
}




int RPCRecordFormatter::unpackBXRecord(const unsigned int* recordIndexInt) {
    
    int bx= ( *recordIndexInt >> rpcraw::bx::BX_SHIFT )& rpcraw::bx::BX_MASK ;
    edm::LogInfo ("RPCUnpacker")<<"Found BX record, BX= "<<bx;
    return bx;

} 


void RPCRecordFormatter::unpackChannelRecord(const unsigned int* recordIndexInt) {

    currentChannel= (*recordIndexInt>> rpcraw::channel::CHANNEL_SHIFT )& rpcraw::channel::CHANNEL_MASK;
    currentRMB=(*recordIndexInt>> rpcraw::channel::TB_RMB_SHIFT)  & rpcraw::channel::TB_RMB_MASK;

    edm::LogInfo ("RPCUnpacker")<<"Found start of Channel Data Record, Channel: "<<currentChannel<<
 	 " Readout Mother Board: "<<currentRMB;
} 

RPCLinkBoardData RPCRecordFormatter::unpackLBRecord(const unsigned int* recordIndexInt) {
    
    int partitionData= (*recordIndexInt>>rpcraw::lb::PARTITION_DATA_SHIFT)&rpcraw::lb::PARTITION_DATA_MASK;
    int halfP = (*recordIndexInt >> rpcraw::lb::HALFP_SHIFT ) & rpcraw::lb::HALFP_MASK;
    int eod = (*recordIndexInt >> rpcraw::lb::EOD_SHIFT ) & rpcraw::lb::EOD_MASK;
    int partitionNumber = (*recordIndexInt >> rpcraw::lb::PARTITION_NUMBER_SHIFT ) & rpcraw::lb::PARTITION_NUMBER_MASK;
    int lbNumber = (*recordIndexInt >> rpcraw::lb::LB_SHIFT ) & rpcraw::lb::LB_MASK ;

    edm::LogInfo ("RPCUnpacker")<< "Found LB Data, LB Number: "<< lbNumber<<
 	" Partition Data "<< partitionData<<
 	" Half Partition " <<  halfP<<
 	" Data Truncated: "<< eod<<
 	" Partition Number " <<  partitionNumber;
    std::vector<int> bits;
    bits.clear();
    for(int bb=0; bb<8;++bb) {
	if ((partitionData>>bb)& 0X1) bits.push_back( partitionNumber* rpcraw::bits::BITS_PER_PARTITION + bb); 
    }
    
    RPCLinkBoardData lbData;
    lbData.setBits(bits);
    lbData.setHalfP(halfP);
    lbData.setEod(eod);
    lbData.setPartitionNumber(partitionNumber);
    lbData.setLbNumber(lbNumber);

    return lbData ;
}



void RPCRecordFormatter::unpackRMBCorruptedRecord(const unsigned int* recordIndexInt,enum RPCRecord::recordTypes type,RPCFEDData & rawData) {
    int channel = (* recordIndexInt>> rpcraw::error::CHANNEL_SHIFT )& rpcraw::error::CHANNEL_MASK;
    int tbRmb   = (* recordIndexInt>> rpcraw::error::TB_RMB_SHIFT)  & rpcraw::error::TB_RMB_MASK;
    if(type==RPCRecord::RMBDiscarded) rawData.addRMBDiscarded(tbRmb, channel);
    if(type==RPCRecord::RMBCorrupted) rawData.addRMBCorrupted(tbRmb, channel);  
 }



