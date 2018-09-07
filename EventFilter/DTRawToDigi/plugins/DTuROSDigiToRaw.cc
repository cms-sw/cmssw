//-------------------------------------------------
//
//   Class: DTuROSDigiToRaw
//
//   L1 DT uROS Raw-to-Digi
//
//
//
//   Author :
//   J. Troconiz  - UAM
//
//
//--------------------------------------------------

#include "EventFilter/DTRawToDigi/plugins/DTuROSDigiToRaw.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "EventFilter/DTRawToDigi/interface/DTROChainCoding.h"
#include "EventFilter/Utilities/interface/DTCRC.h"

#include <iostream>


DTuROSDigiToRaw::DTuROSDigiToRaw(const edm::ParameterSet& pset) :  eventNum(0) {

  produces<FEDRawDataCollection>();

  DTDigiInputTag_ = pset.getParameter<edm::InputTag>("digiColl");

  debug_ = pset.getUntrackedParameter<bool>("debug", false);

  for (int i=FEDNumbering::MINDTUROSFEDID; i<=FEDNumbering::MAXDTUROSFEDID; i++)
        feds_.push_back(i);
 
  nfeds_ = feds_.size();

  Raw_token = consumes<DTDigiCollection>(DTDigiInputTag_);

}


DTuROSDigiToRaw::~DTuROSDigiToRaw(){}


void DTuROSDigiToRaw::produce(edm::Event& e, const edm::EventSetup& c) {

  FEDRawDataCollection data;

  if (!fillRawData(e, c, data)) return;

  auto fed_product = std::make_unique<FEDRawDataCollection>(data);

  e.put(std::move(fed_product));

}


bool DTuROSDigiToRaw::fillRawData(edm::Event& e, const edm::EventSetup& c, FEDRawDataCollection& data) {

  eventNum = e.id().event();

  edm::Handle<DTDigiCollection> digis;
  e.getByToken(Raw_token, digis);

  edm::ESHandle<DTReadOutMapping> mapping;
  c.get<DTReadOutMappingRcd>().get( mapping );

  for (int w_i = 0; w_i < nfeds_; ++w_i) {
    process(feds_[w_i], digis, mapping, data);
  }
  
  return true;
}


void DTuROSDigiToRaw::process(int DTuROSFED,
                              edm::Handle<DTDigiCollection> digis,
                              edm::ESHandle<DTReadOutMapping> mapping,
                              FEDRawDataCollection& data) {


  clear();


  //--> DTDigi analysis

  DTDigiCollection::DigiRangeIterator dttax;
  for (dttax = digis->begin(); dttax != digis->end(); ++dttax){
    const DTDigiCollection::Range& dttar = (*dttax).second;
    const DTLayerId dttal = (*dttax).first;
    for (DTDigiCollection::const_iterator ta = dttar.first; ta != dttar.second; ++ta) {

      int wheelId   = dttal.wheel();
      int sectorId  = dttal.sector();
      int stationId = dttal.station();
      int slId      = dttal.superlayer();
      int layerId   = dttal.layer();
      int cellId    = (*ta).wire();

      int dduId, rosId, robId, tdcId, tdcChannel;
      if ( ! mapping->geometryToReadOut(wheelId, stationId, sectorId, slId, layerId, cellId,
					dduId, rosId, robId, tdcId, tdcChannel)) {

	int crate = theCRT(dduId, rosId);

	if (crate != DTuROSFED) continue;

	int slot  = theSLT(dduId, rosId, robId);
	int link  = theLNK(dduId, rosId, robId);

	int tdcTime   = (*ta).countsTDC();

	bslts[slot-1]++;

	int word = (      (link&0x7F)<<21)
	         + (     (tdcId&0x03)<<19)
	         + ((tdcChannel&0x1F)<<14)
	         + (    tdcTime&0x3FFF   );

	wslts[slot-1].push_back(word);

      }
    }
  }


  int lines = 4;
  int nslts = 0;

  for (int sltit = 0; sltit < DOCESLOTS; sltit++) {

    if (bslts[sltit] == 0) continue;
    nslts += 1;
    lines += 1;

    dslts[sltit] = ((bslts[sltit]+1)/2)+5;
    lines += dslts[sltit];
  }

  FEDRawData& dttfdata = data.FEDData(DTuROSFED);
  dttfdata.resize(lines*8); // size in bytes
  unsigned char* lineFED=dttfdata.data();

  int dataWord1, dataWord2;

  //--> Header

  dataWord1 = 0x50000000
             + (eventNum&0xFFFFFF);
  dataWord2 = (DTuROSFED&0xFFF)<<8;

  int newCRC =  0xFFFF;
  dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

  *((int*)lineFED)=dataWord2; 
  lineFED+=4;
  *((int*)lineFED)=dataWord1; 

  //--> AMC sizes

  dataWord1 = (nslts&0xF)<<20;
  dataWord2 = 0;

  dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

  lineFED+=4;
  *((int*)lineFED)=dataWord2; 
  lineFED+=4;
  *((int*)lineFED)=dataWord1; 

  for (int sltit = 0; sltit < DOCESLOTS; sltit++) {

    if (bslts[sltit] == 0) continue;

    dataWord1 = (dslts[sltit]&0xFFFFFF);
    dataWord2 = ((sltit+1)&0xF)<<16;

    dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 
  }

  //--> AMC data

  for (int sltit = 0; sltit < DOCESLOTS; sltit++) {

    if (bslts[sltit] == 0) continue;

    dataWord1 = ((sltit+1)&0xF)<<24;
    dataWord2 = 0;

    dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    dataWord1 = 0;
    dataWord2 = 0;

    dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    for (int nhit = 0; nhit < bslts[sltit]/2; nhit++) {

      dataWord1 = 0x20000000
                + wslts[sltit].at(nhit*2);
      dataWord2 = wslts[sltit].at(nhit*2+1);

      dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

      lineFED+=4;
      *((int*)lineFED)=dataWord2; 
      lineFED+=4;
      *((int*)lineFED)=dataWord1; 
    }

    if (bslts[sltit]%2 == 1) {

      dataWord1 = 0x20000000
                + wslts[sltit].at(bslts[sltit]-1);
      dataWord2 = 0x1FFFFFFF;

      dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

      lineFED+=4;
      *((int*)lineFED)=dataWord2; 
      lineFED+=4;
      *((int*)lineFED)=dataWord1; 
    }

    dataWord1 = 0x40000000;
    dataWord2 = 0;

    dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    dataWord1 = 0x40000000;
    dataWord2 = 0;

    dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    dataWord1 = 0;
    dataWord2 = (dslts[sltit]&0xFFFFF);

    dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 
  }

  //--> Trailer - line 1

  dataWord1 = 0;
  dataWord2 = 0;

  dt_crc::calcCRC(dataWord1, dataWord2, newCRC);

  lineFED+=4;
  *((int*)lineFED)=dataWord2; 
  lineFED+=4;
  *((int*)lineFED)=dataWord1; 

  //--> Trailer - line 2

  dataWord1 = 0xA0000000
             + (lines&0xFFFFFF);
  dataWord2 = 0;

  dt_crc::calcCRC(dataWord1, dataWord2&0xFFFF, newCRC);

  dataWord2 += (newCRC&0xFFFF)<<16;

  lineFED+=4;
  *((int*)lineFED)=dataWord2; 
  lineFED+=4;
  *((int*)lineFED)=dataWord1; 

  return;
}


void DTuROSDigiToRaw::clear() {

  for (int sltit = 0; sltit < DOCESLOTS; sltit++) {

    bslts[sltit]=0;
    dslts[sltit]=0;
    wslts[sltit].clear();
  }

  return;
}


int DTuROSDigiToRaw::theCRT(int ddu, int ros) {

  if (ros > 6 && ddu > 774) ddu = ddu - 5;
  
  if (ddu == 770) return FEDNumbering::MINDTUROSFEDID;
  else if (ddu == 771) return FEDNumbering::MINDTUROSFEDID;
  else if (ddu == 772) return FEDNumbering::MINDTUROSFEDID+1;
  return FEDNumbering::MAXDTUROSFEDID;
}


int DTuROSDigiToRaw::theSLT(int ddu, int ros, int rob) {

  if (ros > 6 && ddu > 774) ddu = ddu - 5;

  int slot = ((ros-1)/3)+1;
  if (rob == 23) slot = 5;
  if (ddu == 771) slot += 6;
  else if (ddu == 774) slot += 6;
  return slot;
}


int DTuROSDigiToRaw::theLNK(int ddu, int ros, int rob) {

  int link = rob;
  if (rob > 14) link = rob+1;
  if (rob == 24) link = 15;
  link += ((ros-1)%3)*24;
  if (rob == 23) link = ros-1;
  return link;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTuROSDigiToRaw);
