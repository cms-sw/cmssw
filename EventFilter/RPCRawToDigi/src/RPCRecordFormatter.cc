/** \file
 * Implementation of class RPCRecordFormatter
 *
 *  $Date: 2006/10/08 12:36:45 $
 *  $Revision: 1.23 $
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

#include "DataFormats/FEDRawData/interface/FEDRawData.h"



#include <vector>
#include <bitset>
#include <sstream>

using namespace std;
using namespace edm;


RPCRecordFormatter::RPCRecordFormatter(int fedId, const RPCReadOutMapping *r)
 : currentFED(fedId), currentBX(0),currentRMB(0),currentTbLinkInputNumber(0), readoutMapping(r){
}

RPCRecordFormatter::~RPCRecordFormatter(){
}

int RPCRecordFormatter::pack( uint32_t rawDetId, const RPCDigi & digi, int trigger_BX,
                    Record & bxRecord, Record & tbRecord, Record & lbRecord) const
{
   LogDebug("pack:") << " detid: " << rawDetId << " digi: "<<digi;
   int stripInDU = digi.strip();

   // decode digi<->map
   std::pair< ChamberRawDataSpec, LinkBoardChannelCoding>
       rawDataFrame = readoutMapping->rawDataFrame(rawDetId, stripInDU);
   const ChamberRawDataSpec & eleIndex = rawDataFrame.first;
   const LinkBoardChannelCoding & channelCoding = rawDataFrame.second;
   if (eleIndex.dccId != currentFED) return 0;           // return 0 if fedId is not correct 


   // LB record
   int current_BX = trigger_BX+digi.bx();
   this->setBXRecord(bxRecord, current_BX);

   // TB record
   int tbLinkInputNumber = eleIndex.tbLinkInputNum; 
   int rmb = eleIndex.dccInputChannelNum; 
   this->setTBRecord(tbRecord, tbLinkInputNumber, rmb);  

   // LB record
   RPCLinkBoardData lbData;
   lbData.setLbNumber(eleIndex.lbNumInLink);
   lbData.setEod(0);
   lbData.setHalfP(0);
   int channel = channelCoding.channel();
   vector<int> bitsOn; bitsOn.push_back(channel);                         
   lbData.setPartitionNumber( channel/rpcraw::bits::BITS_PER_PARTITION ); 
   lbData.setBits(bitsOn);
   this->setLBRecord(lbRecord, lbData);

   return 1;                                              // return 1 when packing OK 
}

/// Note that it takes a reference to std::auto_ptr<RPCDigiCollection> because
/// I don't want to transfer ownership of RPCDigiCollection (I.S.)

void RPCRecordFormatter::recordUnpack(RPCRecord & theRecord,
		std::auto_ptr<RPCDigiCollection> & prod, RPCFEDData & rawData, int triggerBX){
    
   enum RPCRecord::recordTypes typeOfRecord = theRecord.type();
   const unsigned int* recordIndexInt= theRecord.buf();

   LogDebug("recordUnpack")<<"==> TYPE OF RECORD: "<<typeOfRecord;

   if(typeOfRecord==RPCRecord::StartOfBXData)      {
      LogTrace("")<<"--> HERE StartOfBXDatarecord! ";
	currentBX = this->unpackBXRecord(recordIndexInt);
      rawData.addBXData(currentBX);

    LogTrace("")<<"ORIGINAL    BX: " <<reinterpret_cast<const bitset<16>&>(*recordIndexInt);
    Record myRecord; setBXRecord(myRecord, currentBX); 
    LogTrace("")<<"CORNSTUCTED BX: " <<reinterpret_cast<const bitset<16>&>(myRecord);
    bitset<16> b1 = reinterpret_cast<const bitset<16>&>(*recordIndexInt);
    bitset<16> b2 = reinterpret_cast<const bitset<16>&>(myRecord);
    bool compare = true; for (int i=0; i<=15;i++) compare &= (b1.test(i)==b2.test(i));  
    if (!compare) LogTrace("")<<"PROBLEM IN COMPARE";

   }	   
    
    if(typeOfRecord==RPCRecord::StartOfTbLinkInputNumberData) {
      LogTrace("")<<"--> HERE StartOfTbLinkInputNumberData record! ";
    	currentRMB=0;
	currentTbLinkInputNumber=0;
	this->unpackTbLinkInputRecord(recordIndexInt);
      
      LogTrace("")<<"ORIGINAL    TB: " <<*reinterpret_cast<const bitset<16>*>(recordIndexInt);
      Record myRecord; setTBRecord(myRecord, currentTbLinkInputNumber, currentRMB);
      LogTrace("")<<"CORNSTUCTED TB: " <<*reinterpret_cast<const bitset<16>*>(&myRecord);
    bitset<16> b1 = reinterpret_cast<const bitset<16>&>(*recordIndexInt);
    bitset<16> b2 = reinterpret_cast<const bitset<16>&>(myRecord);
    bool compare = true; for (int i=0; i<=15;i++) compare &= (b1.test(i)==b2.test(i));  
    if (!compare) LogTrace("")<<"PROBLEM IN COMPARE";
    }
   
    /// Unpacking BITS With Hit (uniquely related to strips with hit)
    if(typeOfRecord==RPCRecord::LinkBoardData)	    {
      LogTrace("")<<"--> HERE LinkBoardData record! ";
	RPCLinkBoardData lbData=this->unpackLBRecord(recordIndexInt);

      LogTrace("")<<"ORIGINAL    LB: " <<*reinterpret_cast<const bitset<16>*>(recordIndexInt);
      Record myRecord; setLBRecord(myRecord, lbData); 
      LogTrace("")<<"CORNSTUCTED LB: " <<*reinterpret_cast<const bitset<16>*>(&myRecord);
    bitset<16> b1 = reinterpret_cast<const bitset<16>&>(*recordIndexInt);
    bitset<16> b2 = reinterpret_cast<const bitset<16>&>(myRecord);
    bool compare = true; for (int i=0; i<=15;i++) compare &= (b1.test(i)==b2.test(i));  
    if (!compare) LogTrace("")<<"PROBLEM IN COMPARE";

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
	       RPCReadOutMapping::StripInDetUnit stripInDetUnit=readoutMapping->strip(eleIndex,lbBit);

               // DetUnit
               rawDetId = stripInDetUnit.first;
	       if(!rawDetId) continue;//A dirty fix to avoid crashes. To be FIXED (a geometry should be revised?)
               // stip
               geomStrip = stripInDetUnit.second;
            } 
            catch (cms::Exception & e) {
               edm::LogInfo("RPC unpacker, exception catched, skip digi")<<e.what(); 
	       edm::LogInfo("Values")<< currentRMB<<" "<<currentTbLinkInputNumber<<" "<<lbData.lbNumber();
              continue;
            }



		/// Creating RPC digi
	    RPCDigi digi(geomStrip,currentBX-triggerBX);

      	/// Committing digi to the product
            LogTrace("") << " HERE detector: " << rawDetId<<" digi strip: "<<digi.strip()<<" digi bx: "<<digi.bx();
            LogTrace("") << " ChamberRawDataSpec: " << eleIndex.print(); 
		 prod->insertDigi(RPCDetId(rawDetId),digi);
          }
	
    }
    
    if(typeOfRecord==RPCRecord::RMBDiscarded || typeOfRecord==RPCRecord::RMBCorrupted ) this->unpackRMBCorruptedRecord(recordIndexInt,typeOfRecord,rawData);
    if(typeOfRecord==RPCRecord::RMBDisabled ) this->unpackRMBDisabledRecord(recordIndexInt,typeOfRecord,rawData);
    if(typeOfRecord==RPCRecord::DCCDiscarded) rawData.addDCCDiscarded();
}



void RPCRecordFormatter::setEmptyRecord( Record& record) 
{
 record = (RPCRecord::controlWordFlag << RPCRecord::RECORD_TYPE_SHIFT);
 record |= (RPCRecord::EmptyOrDCCDiscardedFlag << RPCRecord::CONTROL_TYPE_SHIFT);
 record |= (RPCRecord::EmptyWordFlag << RPCRecord::EMPTY_OR_DCCDISCARDED_SHIFT);
}


int RPCRecordFormatter::unpackBXRecord( const unsigned int* recordIndexInt) {
    
    int bx= ( *recordIndexInt >> rpcraw::bx::BX_SHIFT )& rpcraw::bx::BX_MASK ;
    edm::LogInfo ("RPCUnpacker")<<"Found BX record, BX= "<<bx;
    return bx;

} 

void RPCRecordFormatter::setBXRecord( Record& record, int bx) 
{
  record = (RPCRecord::controlWordFlag << RPCRecord::RECORD_TYPE_SHIFT);
  record |= (RPCRecord::BXFlag << RPCRecord::BX_TYPE_SHIFT); 
  record |= (bx << rpcraw::bx::BX_SHIFT);
}

void RPCRecordFormatter::setTBRecord( Record& record, int tbLinkInputNumber, int rmb)
{
  record = (RPCRecord::controlWordFlag << RPCRecord::RECORD_TYPE_SHIFT);
  record |= (RPCRecord::StartOfLBInputDataFlag << RPCRecord::CONTROL_TYPE_SHIFT);
  record |= (tbLinkInputNumber << rpcraw::tb_link::TB_LINK_INPUT_NUMBER_SHIFT);
  record |= (rmb << rpcraw::tb_link::TB_RMB_SHIFT);
}

void RPCRecordFormatter::unpackTbLinkInputRecord(const unsigned int* recordIndexInt) {

    currentTbLinkInputNumber= (*recordIndexInt>> rpcraw::tb_link::TB_LINK_INPUT_NUMBER_SHIFT )& rpcraw::tb_link::TB_LINK_INPUT_NUMBER_MASK;
    currentRMB=(*recordIndexInt>> rpcraw::tb_link::TB_RMB_SHIFT)  & rpcraw::tb_link::TB_RMB_MASK;

    edm::LogInfo ("RPCUnpacker")<<"Found start of LB Link Data Record, tbLinkInputNumber: "<<currentTbLinkInputNumber<<
 	 " Readout Mother Board: "<<currentRMB;
} 


void RPCRecordFormatter::setLBRecord( Record& record, const RPCLinkBoardData & lbData) {
  record = 0;

  int eod = lbData.eod();
  record |= (eod<<rpcraw::lb::EOD_SHIFT );

  int halfP = lbData.halfP();
  record |= (halfP<<rpcraw::lb::HALFP_SHIFT);

  int partitionNumber = lbData.partitionNumber(); 
  record |= (partitionNumber<<rpcraw::lb::PARTITION_NUMBER_SHIFT);

  int lbNumber = lbData.lbNumber();
  record |= (lbNumber<<rpcraw::lb::LB_SHIFT);

  std::vector<int> bitsOn = lbData.bitsOn();
  int partitionData = 0; 
  for (vector<int>::const_iterator iv = bitsOn.begin(); iv != bitsOn.end(); iv++ ) {
    int ibit = (partitionNumber)? (*iv)%(partitionNumber*rpcraw::bits::BITS_PER_PARTITION) : (*iv);
    partitionData |= (1<<ibit); 
  }
  record |= (partitionData<<rpcraw::lb::PARTITION_DATA_SHIFT);
   
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


void RPCRecordFormatter::unpackRMBDisabledRecord(const unsigned int* recordIndexInt,enum RPCRecord::recordTypes type, RPCFEDData & rawData) {
    	int rmbDisabled = (* recordIndexInt>> rpcraw::error::RMB_DISABLED_SHIFT ) & rpcraw::error::RMB_DISABLED_MASK;
	rawData.addRMBDisabled(rmbDisabled);  
    	edm::LogInfo ("RPCUnpacker")<< "Found RMB Disabled: "<<rmbDisabled;
 }

