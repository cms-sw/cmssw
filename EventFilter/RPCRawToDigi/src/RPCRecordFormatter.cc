/** \file
 * Implementation of class RPCRecordFormatter
 *
 *  $Date: 2006/08/08 12:30:36 $
 *  $Revision: 1.17 $
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
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberRawDataSpec.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


#include <vector>

//#define RPCRECORFORMATTER_DEBUG_

RPCRecordFormatter::RPCRecordFormatter(int fedId, const RPCReadOutMapping *r)
 : currentFED(fedId), currentBX(0),currentRMB(0),currentTbLinkInputNumber(0), readoutMapping(r){
}

RPCRecordFormatter::~RPCRecordFormatter(){
}

/// Note that it takes a reference to std::auto_ptr<RPCDigiCollection> because
/// I don't want to transfer ownership of RPCDigiCollection (I.S.)
void RPCRecordFormatter::recordUnpack(RPCRecord & theRecord,
		std::auto_ptr<RPCDigiCollection> & prod, RPCFEDData & rawData){
    
   enum RPCRecord::recordTypes typeOfRecord = theRecord.type();
   const unsigned int* recordIndexInt= theRecord.buf();

   if(typeOfRecord==RPCRecord::StartOfBXData)      {
	currentBX = this->unpackBXRecord(recordIndexInt);
        rawData.addBXData(currentBX);
   }	   
    
    if(typeOfRecord==RPCRecord::StartOfTbLinkInputNumberData) {
    	currentRMB=0;
	currentTbLinkInputNumber=0;
	this->unpackTbLinkInputRecord(recordIndexInt);
    }
   
    /// Unpacking BITS With Hit (uniquely related to strips with hit)
    if(typeOfRecord==RPCRecord::LinkBoardData)	    {
	RPCLinkBoardData lbData=this->unpackLBRecord(recordIndexInt);

      ChamberRawDataSpec eleIndex;
      eleIndex.dccId = currentFED;
      eleIndex.dccInputChannelNum = currentRMB;
      eleIndex.tbLinkInputNum = currentTbLinkInputNumber;
      eleIndex.lbNumInLink = lbData.lbNumber();

	rawData.addRMBData(currentRMB,currentTbLinkInputNumber, lbData);  

      const LinkBoardSpec* linkBoard = readoutMapping->location(eleIndex);

      if (!linkBoard) {
         throw cms::Exception("Invalid Linkboard location!") 
                  << "dccId: "<<eleIndex.dccId
                  << "dccInputChannelNum: " <<eleIndex.dccInputChannelNum
                  << " tbLinkInputNum: "<<eleIndex.tbLinkInputNum
                  << " lbNumInLink: "<<eleIndex.lbNumInLink;
      }

	std::vector<int> bits=lbData.bitsOn();
	for(std::vector<int>::iterator pBit = bits.begin(); pBit !=
    		      bits.end(); ++pBit){

            // fired strip in LB frame
		int lbBit = *(pBit);

            uint32_t rawDetId;
            int geomStrip;
            try {
	      //RPCReadOutMapping::StripInDetUnit stripInDetUnit=linkBoard->strip(lbBit);
	      RPCReadOutMapping::StripInDetUnit stripInDetUnit=readoutMapping->strip(eleIndex,lbBit);

               // DetUnit
               rawDetId = stripInDetUnit.first;
	       if(!rawDetId) continue;//A dirty fix to avoid crashes. To be FIXED (a geometry should be revised?)
               // stip
               geomStrip = stripInDetUnit.second;
            } 
            catch (cms::Exception & e) {
              LogDebug("exception catched, skip digi")<<e.what(); 
              continue;
            }

//          std::cout << " febInLB: " << febInLB 
//                    << " stripPin: " << stripPinInFeb 
//                    << " (partitionNumber: " <<lbData.partitionNumber()
//                    <<" half: " << lbData.halfP()<< " rawBit: "<<rawBit
//                    <<" )"<< std::endl;


		/// Creating RPC digi
		RPCDigi digi(geomStrip,currentBX);

		/// Committing digi to the product
		prod->insertDigi(RPCDetId(rawDetId),digi);
          }
    }
    
    if(typeOfRecord==RPCRecord::RMBDiscarded || typeOfRecord==RPCRecord::RMBCorrupted) this->unpackRMBCorruptedRecord(recordIndexInt,typeOfRecord,rawData);
    if(typeOfRecord==RPCRecord::DCCDiscarded) rawData.addDCCDiscarded();
}




int RPCRecordFormatter::unpackBXRecord(const unsigned int* recordIndexInt) {
    
    int bx= ( *recordIndexInt >> rpcraw::bx::BX_SHIFT )& rpcraw::bx::BX_MASK ;
    #ifdef RPCRECORFORMATTER_DEBUG_
    edm::LogInfo ("RPCUnpacker")<<"Found BX record, BX= "<<bx;
    #endif
    return bx;

} 


void RPCRecordFormatter::unpackTbLinkInputRecord(const unsigned int* recordIndexInt) {

    currentTbLinkInputNumber= (*recordIndexInt>> rpcraw::tb_link::TB_LINK_INPUT_NUMBER_SHIFT )& rpcraw::tb_link::TB_LINK_INPUT_NUMBER_MASK;
    currentRMB=(*recordIndexInt>> rpcraw::tb_link::TB_RMB_SHIFT)  & rpcraw::tb_link::TB_RMB_MASK;

    #ifdef RPCRECORFORMATTER_DEBUG_
    edm::LogInfo ("RPCUnpacker")<<"Found start of LB Link Data Record, tbLinkInputNumber: "<<currentTbLinkInputNumber<<
 	 " Readout Mother Board: "<<currentRMB;
    #endif
} 

RPCLinkBoardData RPCRecordFormatter::unpackLBRecord(const unsigned int* recordIndexInt) {
    
    int partitionData= (*recordIndexInt>>rpcraw::lb::PARTITION_DATA_SHIFT)&rpcraw::lb::PARTITION_DATA_MASK;
    int halfP = (*recordIndexInt >> rpcraw::lb::HALFP_SHIFT ) & rpcraw::lb::HALFP_MASK;
    int eod = (*recordIndexInt >> rpcraw::lb::EOD_SHIFT ) & rpcraw::lb::EOD_MASK;
    int partitionNumber = (*recordIndexInt >> rpcraw::lb::PARTITION_NUMBER_SHIFT ) & rpcraw::lb::PARTITION_NUMBER_MASK;
    int lbNumber = (*recordIndexInt >> rpcraw::lb::LB_SHIFT ) & rpcraw::lb::LB_MASK ;

    #ifdef RPCRECORFORMATTER_DEBUG_
    edm::LogInfo ("RPCUnpacker")<< "Found LB Data, LB Number: "<< lbNumber<<
 	" Partition Data "<< partitionData<<
 	" Half Partition " <<  halfP<<
 	" Data Truncated: "<< eod<<
 	" Partition Number " <<  partitionNumber;
    #endif

    std::vector<int> bits;
    bits.clear();
    for(int bb=0; bb<8;++bb) {
      if(partitionNumber>11){continue;} //Temporasry FIX. Very dirty. AK
	if ((partitionData>>bb)& 0X1) bits.push_back( partitionNumber* rpcraw::bits::BITS_PER_PARTITION + bb);
	}
    
    RPCLinkBoardData lbData(bits,halfP,eod,partitionNumber,lbNumber);

    return lbData ;
}



void RPCRecordFormatter::unpackRMBCorruptedRecord(const unsigned int* recordIndexInt,enum RPCRecord::recordTypes type,RPCFEDData & rawData) {
    int tbLinkInputNumber = (* recordIndexInt>> rpcraw::error::TB_LINK_SHIFT )& rpcraw::error::TB_LINK_MASK;
    int tbRmb   = (* recordIndexInt>> rpcraw::error::TB_RMB_SHIFT)  & rpcraw::error::TB_RMB_MASK;
    if(type==RPCRecord::RMBDiscarded) rawData.addRMBDiscarded(tbRmb, tbLinkInputNumber);
    if(type==RPCRecord::RMBCorrupted) rawData.addRMBCorrupted(tbRmb, tbLinkInputNumber);  
 }



