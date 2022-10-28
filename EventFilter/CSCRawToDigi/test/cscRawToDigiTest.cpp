#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData2007.h"

int main() {
  // For Run-1
  CSCTMBHeader::selfTest(2007, 0x50c3);
  // For Run-2
  CSCTMBHeader::selfTest(2013, 0x6200);
  // For Run-3 (O)TMB firmware and data format

  CSCTMBHeader::selfTest(2020, 0x401);  // OTMB MEx1 CCLUT/HMT Run3 format
  CSCTMBHeader::selfTest(2020, 0x601);  // OTMB MEx1 CCLUT/HMT+GEM Run3 format
  CSCTMBHeader::selfTest(2020, 0x801);  // copper TMB hybrid anode HMT-only CLCT Run2 LCT Run3 format

  CSCTMBHeader::selfTest(2020, 0x021);  // TMB CCLUT/HMT Run2 format
  CSCTMBHeader::selfTest(2020, 0x221);  // OTMB CCLUT/HMT Run2 format
  CSCTMBHeader::selfTest(2020, 0x421);  // OTMB MEx1 2020 Run2 format
  CSCTMBHeader::selfTest(2020, 0x621);  // OTMB ME11 2020 Run2 format

  CSCALCTHeader::selfTest(2007);
  CSCALCTHeader::selfTest(2006);
  CSCTMBData::selfTest();
  CSCTMBTrailer::selfTest();
  //CSCEventData::selfTest();
  CSCComparatorData::selfTest();

  CSCAnodeData2006::selfTest();
  CSCAnodeData2007::selfTest();
}
