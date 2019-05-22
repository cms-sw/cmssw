//-------------------------------------------------
//
//   Class: DTuROSRawToDigi
//
//   L1 DT uROS Raw-to-Digi
//
//
//
//   Author :
//   C. Heidemann - RWTH Aachen
//   J. Troconiz  - UAM
//
//
//--------------------------------------------------

#include "EventFilter/DTRawToDigi/plugins/DTuROSRawToDigi.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

#include "DataFormats/DTDigi/interface/DTuROSControlData.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "EventFilter/DTRawToDigi/interface/DTROChainCoding.h"
#include "EventFilter/Utilities/interface/DTCRC.h"

#include <iostream>


DTuROSRawToDigi::DTuROSRawToDigi(const edm::ParameterSet& pset) {

  produces<DTDigiCollection>();
  produces<std::vector<DTuROSFEDData>>();

  DTuROSInputTag_ = pset.getParameter<edm::InputTag>("inputLabel");

  debug_ = pset.getUntrackedParameter<bool>("debug", false);

  for (int i=FEDNumbering::MINDTUROSFEDID; i<=FEDNumbering::MAXDTUROSFEDID; i++)
  	feds_.push_back(i);
 
  nfeds_ = feds_.size();

  Raw_token = consumes<FEDRawDataCollection>(DTuROSInputTag_);

}


DTuROSRawToDigi::~DTuROSRawToDigi(){}


void DTuROSRawToDigi::produce(edm::Event& e, const edm::EventSetup& c) {

  DTDigiCollection digis;
  std::vector<DTuROSFEDData> words;

  if (!fillRawData(e, c, digis, words)) return;

  auto uROSDTDigi_product = std::make_unique<DTDigiCollection>(digis);
  auto uROSDTWord_product = std::make_unique<std::vector<DTuROSFEDData>>(words);

  e.put(std::move(uROSDTDigi_product));
  e.put(std::move(uROSDTWord_product));

}


bool DTuROSRawToDigi::fillRawData(edm::Event& e, const edm::EventSetup& c,
                                  DTDigiCollection& digis, std::vector<DTuROSFEDData>& words) {

  edm::Handle<FEDRawDataCollection> data;
  e.getByToken(Raw_token, data);

  edm::ESHandle<DTReadOutMapping> mapping;
  c.get<DTReadOutMappingRcd>().get( mapping );

  for (int w_i = 0; w_i < nfeds_; ++w_i) {
    DTuROSFEDData fwords;
    process(feds_[w_i], data, mapping, digis, fwords);
    if (fwords.getfed() >= 0) words.push_back(fwords);
  }
  
  return true;
}


void DTuROSRawToDigi::process(int DTuROSFED,
                              edm::Handle<FEDRawDataCollection> data,
                              edm::ESHandle<DTReadOutMapping> mapping,
                              DTDigiCollection& digis,
                              DTuROSFEDData& fwords) {


  // Container
  std::vector<long> DTuROSWordContainer;


  // Header constituents
  int BOEevTy, DTuROSId;


  // Trailer constituents
  int chkEOE, evtLgth, CRC;


  // Hit counter
  std::map<uint32_t, int> hitOrder;


  //--> Header - line 1


  FEDRawData dturosdata = data->FEDData(DTuROSFED); 
  if ( dturosdata.size() == 0 ) return;


  lineFED=dturosdata.data(); 
  long dataWord = 0;
  int lines  = 0; // counting including header
  readline(lines, dataWord);
  const int posBOEevTy = 60 ; // positions 60 -> 63
  const int posDTuROSId = 8 ; // positions 8 -> 19
  const int posNslost = 52 ; // positions 52 -> 55
  const int posSlotFED = 16 ; // positions 16 -> 19

  BOEevTy  = ( dataWord >> posBOEevTy ) & 0xF;  
  DTuROSId = ( dataWord >> posDTuROSId ) & 0xFFF; 


  if ( (BOEevTy != 0x5) || (DTuROSId != DTuROSFED) ) {
    if ( debug_ ) edm::LogWarning("dturos_unpacker") << "Not a DTuROS FED " << DTuROSFED << " or header " << std::hex << dataWord;
    return;
  }


  fwords.setfed(DTuROSId);
  fwords.setheader1(dataWord);


  int newCRC = 0xFFFF;
  dt_crc::calcCRC(dataWord, newCRC);

  int crate = DTuROSId;


  //--> Header - line 2


  readline(lines, dataWord);
  dt_crc::calcCRC(dataWord, newCRC);

  int nslots = ( dataWord >> posNslost ) & 0xF; 


  fwords.setheader2(dataWord);
  fwords.setnslots(nslots);


  //--> AMC - line 3 to 2+nslots
  std::map<int, int> slot_size;
  for (int j = 0; j < nslots; ++j) {

    readline(lines, dataWord); 
    dt_crc::calcCRC(dataWord, newCRC);

    int slot = ( dataWord >> posSlotFED ) & 0xF; 

    if ( (slot < 1) || (slot > 12) ) {
      if ( debug_ ) edm::LogWarning("dturos_unpacker") << "AMCnumber " << std::dec << slot << " out of range (1-12)";
        return;
    }    


    slot_size[slot] = ( dataWord >> 32 ) & 0xFFFFFF; // positions 32 -> 55


    fwords.setslotsize(slot, slot_size[slot]);

  }

  //--> DTuROS data


  std::map<int,int>::iterator sziterator = slot_size.begin();
  std::map<int,int>::iterator szitend = slot_size.end();  
  for (; sziterator != szitend; ++sziterator) {
    for (int k=0; k<sziterator->second; ++k) {

       readline(lines, dataWord);
       dt_crc::calcCRC(dataWord, newCRC);
       
       DTuROSWordContainer.push_back(dataWord);
    
    }
  }  


  //--> Trailer - line 1


  readline(lines, dataWord);
  dt_crc::calcCRC(dataWord, newCRC);

  //--> Trailer - line 2


  readline(lines, dataWord);

  const int posEOE = 60; // positions 60 -> 63
  const int posEvtLenght = 32; // positions 33 ->55
  const int posCRC = 16; // positions 16 ->31
  chkEOE = ( dataWord >> posEOE ) & 0xF; 


  if ( chkEOE != 0xA ) {
    if ( debug_ )  edm::LogWarning("dturos_unpacker") << "Trailer " << std::hex << dataWord << " does not start with 0xA";
    return;
  }    


  evtLgth = ( dataWord >> posEvtLenght ) & 0xFFFFFF; 
  CRC     = ( dataWord >> posCRC ) & 0xFFFF;   


  dt_crc::calcCRC(dataWord & 0xFFFFFFFF0000FFFF, newCRC);

  if ( newCRC != CRC ) {
    if ( debug_ ) edm::LogWarning("dturos_unpacker") 
      << "Calculated CRC " << std::hex << newCRC << " differs from CRC in trailer " << std::hex << CRC;
    return;
  }


  if ( lines != evtLgth ) {
    if ( debug_ ) edm::LogWarning("dturos_unpacker") 
      << "Number of words read != event length " << std::dec << lines << " " << evtLgth;
    return;
  }


  fwords.settrailer(dataWord);
  fwords.setevtlgth(evtLgth);


  //--> analyze event

 
  std::vector<long>::iterator DTuROSiterator = DTuROSWordContainer.begin();
  std::vector<long>::iterator DTuROSitend = DTuROSWordContainer.end();

  const int posSlot = 56;  // positions 56 -> 59

  for (; DTuROSiterator != DTuROSitend; ++DTuROSiterator) {

    DTuROSROSData rwords;


    dataWord  = (*DTuROSiterator); // Header AMC 1


    int slot   = ( dataWord >> posSlot ) & 0xF;      

    
    if ( (slot < 1) || (slot > 12) ) {
      if ( debug_ ) edm::LogWarning("dturos_unpacker") << "Slot " << std::dec << slot << " out of range (1-12) in crate " << crate;
      break;
    }


    rwords.setslot(slot);
    rwords.setheader1(dataWord);


    ++DTuROSiterator;
    dataWord  = (*DTuROSiterator); // Header AMC 2


    rwords.setheader2(dataWord);
    int slotMap = dataWord & 0xF;
    if (slotMap == 0) slotMap=slot;
    const int posSel1 = 60 ; // positions 60 -> 6
    const int posSel2 = 28 ; // position  28 
    const int posTDCTime = 32 ; // positions  32 -> 45
    const int posTDCChannel = 46 ; // positions  46 -> 50
    const int posTDCId = 51 ; // positions  51 -> 52
    const int posLink = 53 ; // positions  53 -> 59
    const int posTDCChannelSel2Null = 14 ; // positions  14 -> 18
    const int posTDCIdSel2Null = 19 ; // positions  19 -> 20
    const int posLinkSel2Null = 21 ; // positions  21 -> 27
    const int posErrorSel3 = 32 ; // positions  32 -> 60

    for (int k=2; k<slot_size[slot]-1; ++k) {


      ++DTuROSiterator;
      dataWord  = (*DTuROSiterator);
      int selector  = ( dataWord >> posSel1 ) & 0xF;
      int selector2 = ( dataWord >> posSel2 ) & 0x1; 

      if ( selector == 4 ) { // OK word 

	if ( rwords.getokword1() ){ rwords.setokword2(dataWord); }
	else { rwords.setokword1(dataWord); }

      }
      else if ( selector >= 8 && selector <= 13 ) { // OK xword

        rwords.setokxword(selector-8, dataWord);                                              

      }
      else if ( selector == 15 ) { // extra word

        rwords.setexword(dataWord);     

      }
      else {

	if ( selector == 2 ) { // TDC word

	  int tdcTime    = ( dataWord >> posTDCTime ) & 0x3FFF; 
	  int tdcChannel = ( dataWord >> posTDCChannel ) & 0x1F;   
	  int tdcId      = ( dataWord >> posTDCId ) & 0x3;    
	  int link       = ( dataWord >> posLink ) & 0x7F;   

	  int dummy = 0;
	  
	  bool tenDDU = ! mapping->readOutToGeometry(779, 7, 1, 1, 1,
						     dummy, dummy, dummy, 
						     dummy, dummy, dummy);

	  int dduId = theDDU(crate, slotMap, link, tenDDU);
	  int rosId = theROS(slotMap, link);
	  int robId = theROB(slotMap, link);


	  DTROChainCoding channelIndex(dduId, rosId, robId, tdcId, tdcChannel);
	  if (hitOrder.find(channelIndex.getCode()) == hitOrder.end()) hitOrder[channelIndex.getCode()] = 0;
	  else hitOrder[channelIndex.getCode()]++;


	  int wheelId, stationId, sectorId, slId,layerId, cellId;
	  if ( ! mapping->readOutToGeometry(dduId, rosId, robId, tdcId, tdcChannel,
					    wheelId, stationId, sectorId, slId, layerId, cellId)) {

	    DTWireId detId = DTWireId(wheelId, stationId, sectorId, slId, layerId, cellId);
	    int wire = detId.wire();

	    DTDigi digi(wire, tdcTime, hitOrder[channelIndex.getCode()]);
	    digis.insertDigi(detId.layerId(),digi);

	  }

	}
        else if ( selector == 3 ) { // error word

	  if (  debug_ ) edm::LogWarning("dturos_unpacker") << "Error word [" << std::dec << k << "] : " << std::hex << dataWord 
							    << std::dec << " in slot " << slot << " in crate " << crate;

	  int error  = ( dataWord >> posErrorSel3 ) & 0x1FFFFFFF; 
	  rwords.seterror(error);

	}


	if ((dataWord & 0x1FFFFFFF) == 0x1FFFFFFF) continue;

	if ( selector2 == 0 ) { // TDC word

	  int tdcTime    = ( dataWord ) & 0x3FFF;       // positions   0 -> 13
	  int tdcChannel = ( dataWord >> posTDCChannelSel2Null ) & 0x1F;   
	  int tdcId      = ( dataWord >> posTDCIdSel2Null ) & 0x3;    
	  int link       = ( dataWord >> posLinkSel2Null ) & 0x7F;   // positions  21 -> 27


	  if (tdcTime == 16383) continue;

	  int dummy = 0;

	  bool tenDDU = ! mapping->readOutToGeometry(779, 7, 1, 1, 1,
						     dummy, dummy, dummy, 
						     dummy, dummy, dummy);

	  int dduId = theDDU(crate, slotMap, link, tenDDU);
	  int rosId = theROS(slotMap, link);
	  int robId = theROB(slotMap, link);


	  DTROChainCoding channelIndex(dduId, rosId, robId, tdcId, tdcChannel);
	  if (hitOrder.find(channelIndex.getCode()) == hitOrder.end()) hitOrder[channelIndex.getCode()] = 0;
	  else hitOrder[channelIndex.getCode()]++;


	  int wheelId, stationId, sectorId, slId,layerId, cellId;
	  if ( ! mapping->readOutToGeometry(dduId, rosId, robId, tdcId, tdcChannel,
					    wheelId, stationId, sectorId, slId, layerId, cellId)) {

	    DTWireId detId = DTWireId(wheelId, stationId, sectorId, slId, layerId, cellId);
	    int wire = detId.wire();

	    DTDigi digi(wire, tdcTime, hitOrder[channelIndex.getCode()]);
	    digis.insertDigi(detId.layerId(),digi);

	  }

	}
        else if ( selector2 == 1 ) { // error word

	  if (  debug_ ) edm::LogWarning("dturos_unpacker") << "Error word [" << std::dec << k << "] : " << std::hex << dataWord 
							    << std::dec << " in slot " << slot << " in crate " << crate;

	  int error  = ( dataWord ) & 0x1FFFFFFF;       // positions   0 -> 28
	  rwords.seterror(error);

	}

      }

    }


    ++DTuROSiterator;
    dataWord  = (*DTuROSiterator); // Trailer AMC


    rwords.settrailer(dataWord);
    fwords.setuROS(slot, rwords);

  }  // end for-loop container content


  return;
}


int DTuROSRawToDigi::theDDU(int crate, int slot, int link, bool tenDDU) {

  int ros = theROS(slot,link); 
  
  int ddu = 772;

  //if (crate == 1368) { ddu = 775; } 
  ////Needed just in case this FED should be used due to fibers lenght

  if (crate == FEDNumbering::MINDTUROSFEDID) {
    if (slot < 7) 
      ddu = 770;
    else 
      ddu = 771;
  }

  if (crate == (FEDNumbering::MINDTUROSFEDID+1)) { ddu = 772; }
 
  if (crate == FEDNumbering::MAXDTUROSFEDID) {
    if (slot < 7) 
      ddu = 773;
    else 
      ddu = 774;
  }

  if (ros > 6 && tenDDU && ddu < 775)
    ddu += 5;

  return ddu;
}


int DTuROSRawToDigi::theROS(int slot, int link) {

  if (slot%6 == 5) return link+1;

  int ros = (link/24) + 3*(slot%6) - 2;
  return ros;
}


int DTuROSRawToDigi::theROB(int slot, int link) {

  if (slot%6 == 5) return 23;
 
  int rob = link%24;
  if (rob < 15) return rob;
  if (rob == 15) return 24;
  return rob-1;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTuROSRawToDigi);
