#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"

constexpr int activeFEBsForChamberType[11] = {0,7,7,0xf,7,0x7f, 0xf,0x3f,0xf,0x3f,0xf};
constexpr int nTBinsForChamberType[11] = {7,7,7,7,7,7,7,7,7,7,7};


CSCALCTHeader2006::CSCALCTHeader2006(int chamberType) { //constructor for digi->raw packing based on header2006
  // we count from 1 to 10, ME11, ME12, ME13, ME1A, ME21, ME22, ....
  init();
  flag_0 = 0xC;
  flag_1 = 0;
  reserved_1 = 0;
  fifoMode = 1;
  // examiner demands this
  l1aMatch = 1;
  lctChipRead = activeFEBsForChamberType[chamberType];
  activeFEBs = lctChipRead;
  nTBins = nTBinsForChamberType[chamberType];
  ///in order to be able to return header via data()
  //memcpy(theOriginalBuffer, &header2006, header2006.sizeForPacking());

}


void CSCALCTHeader2006::setEventInformation(const CSCDMBHeader & dmb)
{
 l1Acc = dmb.l1a();
 cscID = dmb.dmbID();
 nTBins = 16;
 bxnCount = dmb.bxn();
}


unsigned short CSCALCTHeader2006::nLCTChipRead() const {///header2006 method
  int count = 0;
  for(int i=0; i<7; ++i) {
    if( (lctChipRead>>i) & 1) ++count;
  }
  return count;
}



std::vector<CSCALCTDigi> CSCALCTs2006::ALCTDigis() const
{
  std::vector<CSCALCTDigi> result;
  result.reserve(2);

  CSCALCTDigi digi0(alct0_valid, alct0_quality, alct0_accel,
                    alct0_pattern, alct0_key_wire,
                    alct0_bxn_low|(alct0_bxn_high<<3),1);
  CSCALCTDigi digi1(alct1_valid, alct1_quality, alct1_accel,
                    alct1_pattern, alct1_key_wire,
                    alct1_bxn_low|(alct1_bxn_high<<3),2);
  result.push_back(digi0); result.push_back(digi1);
  return result;
}


void CSCALCTs2006::add(const std::vector<CSCALCTDigi> & digis)
{
  //FIXME doesn't do any sorting
  if(digis.size() > 0) addALCT0(digis[0]);
  if(digis.size() > 1) addALCT1(digis[1]);
}

void CSCALCTs2006::addALCT0(const CSCALCTDigi & digi)
{
  alct0_valid = digi.isValid();
  alct0_quality = digi.getQuality();
  alct0_accel = digi.getAccelerator();
  alct0_pattern = digi.getCollisionB();
  alct0_key_wire = digi.getKeyWG();
  // probably not right
  alct0_bxn_low = digi.getBX();
}


void CSCALCTs2006::addALCT1(const CSCALCTDigi & digi)
{
  alct1_valid = digi.isValid();
  alct1_quality = digi.getQuality();
  alct1_accel = digi.getAccelerator();
  alct1_pattern = digi.getCollisionB();
  alct1_key_wire = digi.getKeyWG();
  // probably not right
  alct1_bxn_low = digi.getBX();
}

