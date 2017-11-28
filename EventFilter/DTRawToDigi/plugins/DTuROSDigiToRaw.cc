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

#include "DTuROSDigiToRaw.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "EventFilter/DTRawToDigi/interface/DTROChainCoding.h"

#include <iostream>


DTuROSDigiToRaw::DTuROSDigiToRaw(const edm::ParameterSet& pset) :  eventNum(0) {

  produces<FEDRawDataCollection>();

  DTDigiInputTag_ = pset.getParameter<edm::InputTag>("DTDigi_Source");

  debug_ = pset.getUntrackedParameter<bool>("debug", false);

  feds_ = pset.getUntrackedParameter<std::vector<int> >("feds", std::vector<int>());
 
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

	int crate = theCRT(dduId);

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
  calcCRC(dataWord1, dataWord2, newCRC);
  *((int*)lineFED)=dataWord2; 
  lineFED+=4;
  *((int*)lineFED)=dataWord1; 

  //--> AMC sizes

  dataWord1 = (nslts&0xF)<<20;
  dataWord2 = 0;

  calcCRC(dataWord1, dataWord2, newCRC);
  lineFED+=4;
  *((int*)lineFED)=dataWord2; 
  lineFED+=4;
  *((int*)lineFED)=dataWord1; 

  for (int sltit = 0; sltit < DOCESLOTS; sltit++) {

    if (bslts[sltit] == 0) continue;

    dataWord1 = (dslts[sltit]&0xFFFFFF);
    dataWord2 = ((sltit+1)&0xF)<<16;

    calcCRC(dataWord1, dataWord2, newCRC);
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

    calcCRC(dataWord1, dataWord2, newCRC);
    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    dataWord1 = 0;
    dataWord2 = 0;

    calcCRC(dataWord1, dataWord2, newCRC);
    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    for (int nhit = 0; nhit < bslts[sltit]/2; nhit++) {

      dataWord1 = 0x20000000
                + wslts[sltit].at(nhit*2);
      dataWord2 = wslts[sltit].at(nhit*2+1);

      calcCRC(dataWord1, dataWord2, newCRC);
      lineFED+=4;
      *((int*)lineFED)=dataWord2; 
      lineFED+=4;
      *((int*)lineFED)=dataWord1; 
    }

    if (bslts[sltit]%2 == 1) {

      dataWord1 = 0x20000000
                + wslts[sltit].at(bslts[sltit]-1);
      dataWord2 = 0x1FFFFFFF;

      calcCRC(dataWord1, dataWord2, newCRC);
      lineFED+=4;
      *((int*)lineFED)=dataWord2; 
      lineFED+=4;
      *((int*)lineFED)=dataWord1; 
    }

    dataWord1 = 0x40000000;
    dataWord2 = 0;

    calcCRC(dataWord1, dataWord2, newCRC);
    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    dataWord1 = 0x40000000;
    dataWord2 = 0;

    calcCRC(dataWord1, dataWord2, newCRC);
    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 

    dataWord1 = 0;
    dataWord2 = (dslts[sltit]&0xFFFFF);

    calcCRC(dataWord1, dataWord2, newCRC);
    lineFED+=4;
    *((int*)lineFED)=dataWord2; 
    lineFED+=4;
    *((int*)lineFED)=dataWord1; 
  }

  //--> Trailer - line 1

  dataWord1 = 0;
  dataWord2 = 0;

  calcCRC(dataWord1, dataWord2, newCRC);
  lineFED+=4;
  *((int*)lineFED)=dataWord2; 
  lineFED+=4;
  *((int*)lineFED)=dataWord1; 

  //--> Trailer - line 2

  dataWord1 = 0xA0000000
             + (lines&0xFFFFFF);
  dataWord2 = 0;

  calcCRC(dataWord1, dataWord2&0xFFFF, newCRC);

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


int DTuROSDigiToRaw::theCRT(int ddu) {

  if (ddu == 770) return 1368;
  else if (ddu == 771) return 1368;
  else if (ddu == 772) return 1369;
  return 1370;
}


int DTuROSDigiToRaw::theSLT(int ddu, int ros, int rob) {

  int slot = ((ros-1)/3)+1;
  if (rob == 23) slot = 5;
  if (ddu == 771) slot += 6;
  else if (ddu == 773) slot += 6;
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


void DTuROSDigiToRaw::calcCRC(int myD1, int myD2, int & myC) {

  int myCRC[16], D[64], C[16];

  for ( int i = 0; i < 32; ++i ) { D[i]    = (myD2>>i) & 0x1; }
  for ( int i = 0; i < 32; ++i ) { D[i+32] = (myD1>>i) & 0x1; }
  for ( int i = 0; i < 16; ++i ) { C[i]    = (myC>>i)  & 0x1; }

  myCRC[0] = ( D[63] + D[62] + D[61] + D[60] + D[55] + D[54] +
               D[53] + D[52] + D[51] + D[50] + D[49] + D[48] +
               D[47] + D[46] + D[45] + D[43] + D[41] + D[40] +
               D[39] + D[38] + D[37] + D[36] + D[35] + D[34] +
               D[33] + D[32] + D[31] + D[30] + D[27] + D[26] +
               D[25] + D[24] + D[23] + D[22] + D[21] + D[20] +
               D[19] + D[18] + D[17] + D[16] + D[15] + D[13] +
               D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  +
               D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
               D[0]  + C[0]  + C[1]  + C[2]  + C[3]  + C[4]  +
               C[5]  + C[6]  + C[7]  + C[12] + C[13] + C[14] +
               C[15] )%2;

  myCRC[1] = ( D[63] + D[62] + D[61] + D[56] + D[55] + D[54] +
	       D[53] + D[52] + D[51] + D[50] + D[49] + D[48] +
	       D[47] + D[46] + D[44] + D[42] + D[41] + D[40] +
	       D[39] + D[38] + D[37] + D[36] + D[35] + D[34] +
    	       D[33] + D[32] + D[31] + D[28] + D[27] + D[26] +
    	       D[25] + D[24] + D[23] + D[22] + D[21] + D[20] +
    	       D[19] + D[18] + D[17] + D[16] + D[14] + D[13] +
    	       D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  +
	       D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
	       C[0]  + C[1]  + C[2]  + C[3]  + C[4]  + C[5]  +
	       C[6]  + C[7]  + C[8]  + C[13] + C[14] + C[15] )%2;

  myCRC[2] = ( D[61] + D[60] + D[57] + D[56] + D[46] + D[42] +
	       D[31] + D[30] + D[29] + D[28] + D[16] + D[14] +
	       D[1]  + D[0]  + C[8]  + C[9]  + C[12] + C[13] )%2;

  myCRC[3] = ( D[62] + D[61] + D[58] + D[57] + D[47] + D[43] +
	       D[32] + D[31] + D[30] + D[29] + D[17] + D[15] +
	       D[2]  + D[1]  + C[9]  + C[10] + C[13] + C[14] )%2;

  myCRC[4] = ( D[63] + D[62] + D[59] + D[58] + D[48] + D[44] +
    	       D[33] + D[32] + D[31] + D[30] + D[18] + D[16] + 
	       D[3]  + D[2]  + C[0]  + C[10] + C[11] + C[14] +
	       C[15] )%2;

  myCRC[5] = ( D[63] + D[60] + D[59] + D[49] + D[45] + D[34] +
	       D[33] + D[32] + D[31] + D[19] + D[17] + D[4]  +
    	       D[3]  + C[1]  + C[11] + C[12] + C[15] )%2;

  myCRC[6] = ( D[61] + D[60] + D[50] + D[46] + D[35] + D[34] +
	       D[33] + D[32] + D[20] + D[18] + D[5]  + D[4]  +
	       C[2]  + C[12] + C[13] )%2;

  myCRC[7] = ( D[62] + D[61] + D[51] + D[47] + D[36] + D[35] +
    	       D[34] + D[33] + D[21] + D[19] + D[6]  + D[5]  +
	       C[3]  + C[13] + C[14] )%2;

  myCRC[8] = ( D[63] + D[62] + D[52] + D[48] + D[37] + D[36] +
	       D[35] + D[34] + D[22] + D[20] + D[7]  + D[6]  +
    	       C[0]  + C[4]  + C[14] + C[15] )%2;

  myCRC[9] = ( D[63] + D[53] + D[49] + D[38] + D[37] + D[36] +
	       D[35] + D[23] + D[21] + D[8]  + D[7]  + C[1]  +
	       C[5]  + C[15] )%2;

  myCRC[10] = ( D[54] + D[50] + D[39] + D[38] + D[37] + D[36] + 
       		D[24] + D[22] + D[9]  + D[8]  + C[2]  + C[6] )%2;

  myCRC[11] = ( D[55] + D[51] + D[40] + D[39] + D[38] + D[37] +
		D[25] + D[23] + D[10] + D[9]  + C[3]  + C[7] )%2;

  myCRC[12] = ( D[56] + D[52] + D[41] + D[40] + D[39] + D[38] +
        	D[26] + D[24] + D[11] + D[10] + C[4]  + C[8] )%2;

  myCRC[13] = ( D[57] + D[53] + D[42] + D[41] + D[40] + D[39] +
	        D[27] + D[25] + D[12] + D[11] + C[5]  + C[9] )%2;

  myCRC[14] = ( D[58] + D[54] + D[43] + D[42] + D[41] + D[40] +
        	D[28] + D[26] + D[13] + D[12] + C[6]  + C[10] )%2;

  myCRC[15] = ( D[63] + D[62] + D[61] + D[60] + D[59] + D[54] +
		D[53] + D[52] + D[51] + D[50] + D[49] + D[48] + 
	        D[47] + D[46] + D[45] + D[44] + D[42] + D[40] +
        	D[39] + D[38] + D[37] + D[36] + D[35] + D[34] + 
		D[33] + D[32] + D[31] + D[30] + D[29] + D[26] +
        	D[25] + D[24] + D[23] + D[22] + D[21] + D[20] + 
        	D[19] + D[18] + D[17] + D[16] + D[15] + D[14] +
        	D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  + 
        	D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
        	D[0]  + C[0]  + C[1]  + C[2]  + C[3]  + C[4]  + 
	        C[5]  + C[6]  + C[11] + C[12] + C[13] + C[14] +
	        C[15] )%2;

  int tempC = 0x0;  
  for ( int i = 0; i < 16 ; ++i) { tempC = tempC + ( myCRC[i] << i ); }
  myC = tempC;
  return;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DTuROSDigiToRaw);
