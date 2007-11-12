#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>
#include <bitset>

bool CSCALCTHeader::debug=false;


CSCALCTHeader::CSCALCTHeader(int chamberType) 
{
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

CSCALCTHeader::CSCALCTHeader(const unsigned short * buf) 
{
  memcpy(this, buf, sizeInWords()*2);
  //printf("%04x %04x %04x %04x\n",buf[4],buf[5],buf[6],buf[7]);
}

CSCALCTHeader::CSCALCTHeader(const CSCALCTStatusDigi & digi)
{
  memcpy(this, digi.header(), sizeInWords()*2);
}





void CSCALCTHeader::setEventInformation(const CSCDMBHeader & dmb) 
{
  l1Acc = dmb.l1a();
  cscID = dmb.dmbID();
  nTBins = 16;
  bxnCount = dmb.bxn();
}

unsigned short CSCALCTHeader::nLCTChipRead() const 
{
  int count = 0;
  for(int i=0; i<7; ++i) 
    {
      if( (lctChipRead>>i) & 1) ++count;
    }
  return count;
}

int CSCALCTHeader::ALCTCRCcalc() 
{
  std::vector< std::bitset<16> > theTotalALCTData;
  int nTotalLines = sizeInWords()+nLCTChipRead()*NTBins()*6*2;
  theTotalALCTData.reserve(nTotalLines);
  for (int line=0; line<nTotalLines; ++line) 
    {
      theTotalALCTData[line] = std::bitset<16>(theOriginalBuffer[line]);
    }

  if ( theTotalALCTData.size() > 0 ) 
    {
      std::bitset<22> CRC=calCRC22(theTotalALCTData);
      return CRC.to_ulong();
    } 
  else 
    {
      edm::LogWarning ("CSCALCTHeader") << "theTotalALCTData doesn't exist";
      return 0;
    }
}

std::vector<CSCALCTDigi> CSCALCTHeader::ALCTDigis() const 
{ 

  std::vector<CSCALCTDigi> result;
  
  //for the zeroth ALCT word:  
  if (debug) 
    edm::LogInfo("CSCALCTHeader") << "ALCT DIGI 1 valid = " << alct0Valid() 
				  << "  quality = "  << alct0Quality()
				  << "  accel = " << alct0Accel()
				  << "  pattern = " << alct0Pattern() 
				  << "  Key Wire Group = " << alct0KeyWire() 
				  << "  BX = " << alct0BXN();  

  CSCALCTDigi digi(alct0Valid(), alct0Quality(), alct0Accel(), alct0Pattern(),
		   alct0KeyWire(), alct0BXN(), 1);
  digi.setFullBX(BXNCount());
  result.push_back(digi);

  //for the first ALCT word:  
  if (debug) 
    edm::LogInfo("CSCALCTHeader") << "ALCT DIGI 2 valid = " << alct1Valid() 
				  << "  quality = "  << alct1Quality()
				  << "  accel = " << alct1Accel()
				  << "  pattern = " << alct1Pattern() 
				  << "  Key Wire Group = " << alct1KeyWire() 
				  << "  BX = " << alct1BXN();  


  digi = CSCALCTDigi(alct1Valid(), alct1Quality(), alct1Accel(), alct1Pattern(),
		     alct1KeyWire(), alct1BXN(), 2);
  digi.setFullBX(BXNCount());
  result.push_back(digi);
  return result;
}


std::bitset<22> CSCALCTHeader::calCRC22(const std::vector< std::bitset<16> >& datain)
{
  std::bitset<22> CRC;
  CRC.reset();
  for(int i=0;i<(int) datain.size();++i)
    {
      if (debug) edm::LogInfo ("CSCALCTHeader") << std::ios::hex << datain[i].to_ulong();
      CRC=nextCRC22_D16(datain[i],CRC);
    }
  return CRC;
}


std::bitset<22> CSCALCTHeader::nextCRC22_D16(const std::bitset<16>& D, 
				       const std::bitset<22>& C)
{
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

std::ostream & operator<<(std::ostream & os, const CSCALCTHeader & header) 
{
  os << "ALCT HEADER CSCID " << header.CSCID()
     << "  L1ACC " << header.L1Acc() << std::endl;
  os << "# ALCT chips read : "  << header.nLCTChipRead() 
     << " time samples " << header.NTBins() << std::endl;
  return os;
}
