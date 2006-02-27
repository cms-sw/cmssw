#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>
#include <bitset>

bool CSCALCTHeader::debug=false;


CSCALCTHeader::CSCALCTHeader(int chamberType) {
  // we count from 1 to 10, ME11, ME12, ME13, ME1A, ME21, ME22, ....
  //static int nAFEBsForChamberType[11] = {0,3,3,3,3,7,4,6,4,6,4};
  // same numbers as above, with one bit for every board
  static int activeFEBsForChamberType[11] = {0,7,7,7,7,0x7f, 0xf,0x3f,0xf,0x3f,0xf};
  static int nTBinsForChamberType[11] = {7,7,7,7,7,7,7,7,7,7,7};

  bzero(this, sizeInWords()*2); 
  flag_0 = 0xC;
  lctChipRead = activeFEBsForChamberType[chamberType];
  activeFEBs = lctChipRead;
  nTBins = nTBinsForChamberType[chamberType];
  if (debug)
    edm::LogInfo ("CSCALCTHeader") << "MAKING ALCTHEADER " << chamberType 
				   << " " << activeFEBs << " " << nTBins;
}

CSCALCTHeader::CSCALCTHeader(const unsigned short * buf) {
  memcpy(this, buf, sizeInWords()*2);
  //printf("%04x %04x %04x %04x\n",buf[4],buf[5],buf[6],buf[7]);
}

void CSCALCTHeader::setEventInformation(const CSCDMBHeader & dmb) {
  l1Acc = dmb.l1a();
  cscID = dmb.dmbID();
  nTBins = 8;
  bxnCount = dmb.bxn();
}

unsigned short CSCALCTHeader::nLCTChipRead() const {
  int count = 0;
  for(int i=0; i<7; ++i) {
    if( (lctChipRead>>i) & 1) ++count;
  }
  return count;
}

int CSCALCTHeader::ALCTCRCcalc() {
  std::vector< std::bitset<16> > theTotalALCTData;
  int nTotalLines = sizeInWords()+nLCTChipRead()*NTBins()*6*2;
  theTotalALCTData.reserve(nTotalLines);
  for (int line=0; line<nTotalLines; line++) {
    theTotalALCTData[line] = std::bitset<16>(theOriginalBuffer[line]);
  }

  if ( theTotalALCTData.size() > 0 ) {
    std::bitset<22> CRC=calCRC22(theTotalALCTData);
    return CRC.to_ulong();
  } else {
    edm::LogWarning ("CSCALCTHeader") << "theTotalALCTData doesn't exist";
    return 0;
  }
}

std::vector<CSCALCTDigi> CSCALCTHeader::ALCTDigis() const { 
  int keyWireGroup;
  int BXCounter;
  int quality;
  int pattern;
  int accel;
  int valid;
  std::vector<CSCALCTDigi> result;
  
  //for the zeroth ALCT word:  
  unsigned int alct0 = alct0Word();
  valid =        alct0 & 0x1;          //(bin:                1)
  quality =      (alct0 & 0x6)>>1;     //(bin:              110) 
  accel =        (alct0 & 0x8)>>3;     //(bin:             1000) 
  pattern =      (alct0 & 0x10)>>4;    //(bin:            10000) 
  keyWireGroup = (alct0 & 0xfe0)>>5;   //(bin:     111111100000)
  BXCounter =    (alct0 & 0x1f000)>>12;//(bin:11111000000000000)
  
  if (debug) edm::LogInfo("CSCALCTHeader") << "ALCT DIGI 0 valid = " << valid 
					   << "  quality = "  << quality
					   << "  accel = " << accel
					   << "  pattern = " << pattern 
					   << "  Key Wire Group = " << keyWireGroup 
					   << "  BX = " << BXCounter;  

  CSCALCTDigi digi(1, keyWireGroup, BXCounter, accel, quality, pattern, valid);
  result.push_back(digi);

  //for the first ALCT word:  
  unsigned int alct1 = alct1Word();
  valid =        alct1 & 0x1;          //(bin:                1)
  quality =      (alct1 & 0x6)>>1;     //(bin:              110) 
  accel =        (alct1 & 0x8)>>3;     //(bin:             1000) 
  pattern =      (alct1 & 0x10)>>4;    //(bin:            10000) 
  keyWireGroup = (alct1 & 0xfe0)>>5;   //(bin:     111111100000)
  BXCounter =    (alct1 & 0x1f000)>>12;//(bin:11111000000000000)
 
  
  if (debug) edm::LogInfo("CSCALCTHeader") << "ALCT DIGI 1 valid = " << valid 
					   << "  quality = " << quality 
					   << "  accel = " << accel
					   << "  pattern = " << pattern 
					   << "  Key Wire Group = " << keyWireGroup 
					   << "  BX = " << BXCounter;
  
  digi = CSCALCTDigi(2, keyWireGroup, BXCounter, accel, quality, pattern, valid);
  result.push_back(digi);
  return result;

}


std::bitset<22> CSCALCTHeader::calCRC22(const std::vector< std::bitset<16> >& datain){
  std::bitset<22> CRC;
  CRC.reset();
  for(int i=0;i<(int) datain.size();i++){
    if (debug) edm::LogInfo ("CSCALCTHeader") << std::ios::hex << datain[i].to_ulong();
    CRC=nextCRC22_D16(datain[i],CRC);
  }
  return CRC;
}


std::bitset<22> CSCALCTHeader::nextCRC22_D16(const std::bitset<16>& D, 
				       const std::bitset<22>& C){
  std::bitset<22> NewCRC;
  
  NewCRC[ 0] = D[ 0] ^ C[ 6];
  NewCRC[ 1] = D[ 1] ^ D[ 0] ^ C[ 6] ^ C[ 7];
  NewCRC[ 2] = D[ 2] ^ D[ 1] ^ C[ 7] ^ C[ 8];
  NewCRC[ 3] = D[ 3] ^ D[ 2] ^ C[ 8] ^ C[ 9];
  NewCRC[ 4] = D[ 4] ^ D[ 3] ^ C[ 9] ^ C[10];
  NewCRC[ 5] = D[ 5] ^ D[ 4] ^ C[10] ^ C[11];
  NewCRC[ 6] = D[ 6] ^ D[ 5] ^ C[11] ^ C[12];
  NewCRC[ 7] = D[ 7] ^ D[ 6] ^ C[12] ^ C[13];
  NewCRC[ 8] = D[ 8] ^ D[ 7] ^ C[13] ^ C[14];
  NewCRC[ 9] = D[ 9] ^ D[ 8] ^ C[14] ^ C[15];
  NewCRC[10] = D[10] ^ D[ 9] ^ C[15] ^ C[16];
  NewCRC[11] = D[11] ^ D[10] ^ C[16] ^ C[17];
  NewCRC[12] = D[12] ^ D[11] ^ C[17] ^ C[18];
  NewCRC[13] = D[13] ^ D[12] ^ C[18] ^ C[19];
  NewCRC[14] = D[14] ^ D[13] ^ C[19] ^ C[20];
  NewCRC[15] = D[15] ^ D[14] ^ C[20] ^ C[21];
  NewCRC[16] = D[15] ^ C[ 0] ^ C[21];
  NewCRC[17] = C[ 1];
  NewCRC[18] = C[ 2];
  NewCRC[19] = C[ 3];
  NewCRC[20] = C[ 4];
  NewCRC[21] = C[ 5];
  
  return NewCRC;
}

std::ostream & operator<<(std::ostream & os, const CSCALCTHeader & header) {
  os << "ALCT HEADER CSCID " << header.CSCID()
     << "  L1ACC " << header.L1Acc() << std::endl;
  os << "# ALCT chips read : "  << header.nLCTChipRead() 
     << " time samples " << header.NTBins() << std::endl;
  return os;
}
