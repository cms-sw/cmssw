/** \file
 * Implementation of class RPCRecordFormatter
 *
 *  $Date: 2006/03/31 07:47:18 $
 *  $Revision: 1.7 $
 *
 * \author Ilaria Segoni
 */

#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecord.h"
#include "EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRawDataPattern.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

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
	int dccId=790;//fedNumber;
	int tbId=currentRMB;
	int lboxId=currentChannel/5;
	int mbId=currentChannel%5;
	int lboardId=lbData.lbNumber();
	
	 rawData.addRMBData(currentRMB,currentChannel, lbData);  

	std::vector<int> bits=lbData.bitsOn();
	for(std::vector<int>::iterator pBit = bits.begin(); pBit !=
    		      bits.end(); ++pBit){

		int bit = *(pBit);
		RPCReadOutMapping rmap;
		int region;
		int ring;
		int station;
		int sector; 
		int layer;
		int subsector;
		int roll;
		int strip;
		rmap.readOutToGeometry(dccId,tbId,lboxId,mbId,lboardId,bit,
				       region,ring,station,sector,layer,
				       subsector,roll,strip);

		RPCDetId detId(region,ring,station,sector,
			       layer,subsector,roll);

		/// Creating RPC digi
		/// When channel mapping available calculate strip
		///and replace bit with strip
		RPCDigi digi(strip,bx);

		/// Committing digi to the product
		prod->insertDigi(detId,digi);
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



