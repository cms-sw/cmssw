//_________________________________________________________
//
//  CSCTMBData 9/18/03  B.Mohr                             
//  Unpacks and collects all TMB data
//_________________________________________________________
//
//THere are serious problems with this class!!!
//findline() gets confused with random data looking like the markers
//need to fix A.Tumanov
//

#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBScope.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <bitset>
#include <cstdio>
#include <boost/dynamic_bitset.hpp>

bool CSCTMBData::debug =false;

CSCTMBData::CSCTMBData() 
  : theOriginalBuffer(0), theTMBScopeIsPresent(false), theTMBScope(0),
theRPCDataIsPresent(false) {
}


CSCTMBData::CSCTMBData(unsigned short *buf) 
  : theOriginalBuffer(buf), theTMBScopeIsPresent(false), theTMBScope(0), theRPCDataIsPresent(false)
{
  size_ = UnpackTMB(buf);

} //CSCTMBData

CSCTMBData::~CSCTMBData() {
  if (theTMBScopeIsPresent) {
    delete theTMBScope;
    theTMBScopeIsPresent = false;
  }
}

// returns -1 if not found
//
//
int findLine(unsigned short *buf, unsigned short marker,int first,  int maxToDo) {
 
  for(int i = first; i < maxToDo; ++i) {
 
    if(buf[i] == marker)  {
	return i;
    }
    
  }

  return -1;

}
  
int CSCTMBData::TMBCRCcalc() {

  std::vector<std::bitset<16> > theTotalTMBData(theE0FLine+1-theB0CLine);

  unsigned i = 0;
  for (unsigned int line=theB0CLine; line<theE0FLine+1;line++) {
    theTotalTMBData[i] = std::bitset<16>(theOriginalBuffer[line]);
    ++i;
  }


  if ( theTotalTMBData.size() > 0 ) {
    std::bitset<22> CRC=calCRC22(theTotalTMBData);
    edm::LogInfo("CSCTMBData") << " Test here " << CRC.to_ulong();
    return CRC.to_ulong();
  } else {
    edm::LogInfo("CSCTMBData") << "theTotalTMBData doesn't exist";
    return 0;
  }
}

int CSCTMBData::UnpackTMB(unsigned short *buf) {

  int b0cLine = findLine(buf, 0x6b0c, 0, 10);
  if(b0cLine == -1) {
    edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: No b0c line!";
    return 0;  
  }
  //
  int Ntbins =  buf[b0cLine+1]&0x1f ;
  int NHeaderFrames = buf[b0cLine+4]&0x1f;
  //
  int TotTMBReadout = 27+Ntbins*6*5+1+Ntbins*2*4+2+8*256+8; //see tmb2004 manual (version v2p06) page54.
  int MaxSizeRPC = 1+Ntbins*2*4+1;
  int MaxSizeScope = 5;
  int e0bLine = findLine(buf, 0x6e0b, 0, TotTMBReadout);
  

  if(e0bLine == -1) {
    edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: No e0b line!";
    edm::LogError("CSCTMBData") << "+++ Corrupt header!";
    return 0;
  } 

  memcpy(&theTMBHeader, buf, NHeaderFrames*2);

  if(!theTMBHeader.check()) {
     edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: Bad TMB header e0bLine=" << std::hex << buf[e0bLine];
     return 0;
  }

  int afterHeader = theTMBHeader.sizeInWords();

  theCLCTData = CSCCLCTData(theTMBHeader.NCFEBs(), theTMBHeader.NTBins(), buf+e0bLine+1);

  if(!theCLCTData.check()) {
     edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: Bad CLCT data";
  }else{
     afterHeader+=theCLCTData.sizeInWords();
  }

  // look for RPC
  int b04Line = findLine(buf, 0x6b04, afterHeader, afterHeader+MaxSizeRPC);
  if(b04Line != -1 ) {
    // we need an e04 line to calculate the size
    int e04Line = findLine(buf, 0x6e04, afterHeader, afterHeader+MaxSizeRPC);
    if(e04Line != -1) {
      if (e04Line < b04Line ) {
	edm::LogError("CSCTMBData") << "RPC data is corrupt! e04Line < b04Line ";
	return 0;
      }
      else {
	theRPCDataIsPresent = true;
	theRPCData = CSCRPCData(buf+b04Line, e04Line-b04Line+1);
	afterHeader+=theRPCData.sizeInWords();
      }
    } else {
      edm::LogError("CSCTMBData") << "CSCTMBData::corrupt RPC data! Failed to find end! ";
      return 0;
    }
  }

 
  // look for scope.  Should there be a 6?
  int b05Line = findLine(buf, 0x6b05, afterHeader, MaxSizeScope);
  if(b05Line != -1) {
     edm::LogInfo("CSCTMBData") <<"found scope!!!!!!!!!!!!!"; 
     int e05Line = findLine(buf, 0x6e05, afterHeader, TotTMBReadout-afterHeader);
     if(e05Line != -1) {     
       theTMBScopeIsPresent = true;
       theTMBScope = new CSCTMBScope(buf,b05Line, e05Line);
       afterHeader+=theTMBScope->sizeInWords();
     }

  }

  int maxLine = findLine(buf, 0xde0f, afterHeader, TotTMBReadout-afterHeader);
  if(maxLine == -1) {
    edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: No e0f line!";
    return 0;
  }

  //Now for CRC check put this information into bitset

  theB0CLine = b0cLine;
  theE0FLine = maxLine;

  int CRClow  = buf[maxLine-2] & 0x7ff;
  int CRChigh = buf[maxLine-1] & 0x7ff;


  CRCCnt = (CRChigh<<11) | (CRClow);

  // finally, the trailer
    int e0cLine = findLine(buf, 0x6e0c, afterHeader, maxLine);
  if (e0cLine == -1) {
    edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: No e0c line!";
  } else {
    theTMBTrailer = CSCTMBTrailer(buf+e0cLine);
  }

  checkSize();

  // size, since we count from 0 and have one more trailer word
  // there are sometimes multiple "de0f" lines in trailer, so key on "6e0c"
  return e0cLine-b0cLine+theTMBTrailer.sizeInWords();
} //UnpackTMB

bool CSCTMBData::checkSize() const {
  // sum up all the components and see if they have the size indicated in the TMBTrailer
  return true;
}

std::bitset<22> CSCTMBData::calCRC22(const std::vector< std::bitset<16> >& datain){
  std::bitset<22> CRC;
  CRC.reset();
  for(unsigned int i=0;i<datain.size()-3;i++){
    CRC=nextCRC22_D16(datain[i],CRC);
  }
  return CRC;
}

CSCTMBScope & CSCTMBData::tmbScope() const {
  if (!theTMBScopeIsPresent) throw("No TMBScope in this chamber");
  return * theTMBScope;
}


std::bitset<22> CSCTMBData::nextCRC22_D16(const std::bitset<16>& D, 
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


boost::dynamic_bitset<> CSCTMBData::pack() {
  boost::dynamic_bitset<> result(theTMBHeader.sizeInWords()*16,*(const unsigned*)&theTMBHeader);
  boost::dynamic_bitset<> clctData(theCLCTData.sizeInWords()*16, (const unsigned)*theCLCTData.data());
  result = append(result,clctData);
  int finalSize = result.size()/16 + theTMBTrailer.sizeInWords(); //size() returns # of bits 
                                                                
  theTMBTrailer.setWordCount(finalSize);
  boost::dynamic_bitset<> tmbTrailer( theTMBTrailer.sizeInWords()*16, *(const unsigned*)&theTMBTrailer);
  result = append(result,tmbTrailer);
  return result;
}

boost::dynamic_bitset<> 
CSCTMBData::append(const boost::dynamic_bitset<> & bs1, const boost::dynamic_bitset<> & bs2)
{
  boost::dynamic_bitset<> result(bs1.size()+bs2.size());
  unsigned size1 = bs1.size();
  for(unsigned i = 0; i < size1; ++i)
    {
      result[i] = bs1[i];
    }
  for(unsigned i = 0; i < bs2.size(); ++i)
    {
      result[size1+i] = bs2[i];
    }
  return result;
}
