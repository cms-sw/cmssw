/** \class CSCTMBData
 *
 *  $Date: 2007/10/17 18:23:48 $
 *  $Revision: 1.18 $
 *  \author A. Tumanov - Rice
 */

#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cstdio>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"

bool CSCTMBData::debug =false;

CSCTMBData::CSCTMBData() 
  : theOriginalBuffer(0), theTMBScopeIsPresent(false), theTMBScope(0),
theRPCDataIsPresent(false) {}


CSCTMBData::CSCTMBData(unsigned short *buf) 
  : theOriginalBuffer(buf), theTMBScopeIsPresent(false), theTMBScope(0), theRPCDataIsPresent(false){
  size_ = UnpackTMB(buf);
} 

CSCTMBData::~CSCTMBData(){
  if (theTMBScopeIsPresent) {
    delete theTMBScope;
    theTMBScopeIsPresent = false;
  }
}

/// returns -1 if not found
/// obsolete
int findLine(unsigned short *buf, unsigned short marker,int first,  int maxToDo) {
  for(int i = first; i < maxToDo; ++i) { 
    if(buf[i] == marker) {
      return i;
    }
  }
  return -1;
}
  
int CSCTMBData::TMBCRCcalc() {
  std::vector<std::bitset<16> > theTotalTMBData(theE0FLine+1-theB0CLine);
  unsigned i = 0;
  for (unsigned int line=theB0CLine; line<theE0FLine+1;++line) {
    theTotalTMBData[i] = std::bitset<16>(theOriginalBuffer[line]);
    ++i;
  }
  if ( theTotalTMBData.size() > 0 )   {
    std::bitset<22> CRC=calCRC22(theTotalTMBData);
    edm::LogInfo("CSCTMBData") << " Test here " << CRC.to_ulong();
    return CRC.to_ulong();
  } 
  else {
    edm::LogInfo("CSCTMBData") << "theTotalTMBData doesn't exist";
    return 0;
  }
}

int CSCTMBData::UnpackTMB(unsigned short *buf) {
  ///determine 2007 or 2006 version
  unsigned short int firmwareVersion=0;
  int Ntbins = 0 ;
  int NHeaderFrames = 0;
  
  int b0cLine=0;///assumes that buf starts at the tmb data
                ///this is not true if something is wrong in the data 
                ///before TMB - then we skip the whole event


  if (buf[b0cLine]==0xdb0c) {
    firmwareVersion=2007;
    Ntbins = buf[b0cLine+19]&0xF8;
    NHeaderFrames = buf[b0cLine+5]&0x3F;
  } 
  else if (buf[b0cLine]==0x6b0c) {
    firmwareVersion=2006;
    Ntbins =  buf[b0cLine+1]&0x1f ;
    NHeaderFrames = buf[b0cLine+4]&0x1f;
  } 
  else {
    edm::LogError("CSCTMBData") << "+++ Can't find b0C flag";
  }

  if ((firmwareVersion==2007)&&(!(((buf[b0cLine]&0xFFFF)==0xDB0C)&&((buf[b0cLine+1]&0xf000)==0xD000)
	&&((buf[b0cLine+2]&0xf000)==0xD000)&&((buf[b0cLine+3]&0xf000)==0xD000)))){
    edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: error in header in 2007 format!";
  }

  int MaxSizeRPC = 1+Ntbins*2*4+1;
  //int MaxSizeScope = 5;
  int e0bLine =-1;
  switch (firmwareVersion) {
  case 2007:
    e0bLine = 42; //last word of header2007
    break;
  case 2006:
    e0bLine = 26; //last word of header in 2006 format
    break;
  default:
    edm::LogError("CSCTMBData") << "+++ undetermined firmware format - cant find e0bLine";
  }

  theTMBHeader=CSCTMBHeader(buf);
  
  if(!theTMBHeader.check())   {
    edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: Bad TMB header e0bLine=" << std::hex << buf[e0bLine];
    return 0;
  }

  int currentPosition = theTMBHeader.sizeInWords();

  theCLCTData = CSCCLCTData(theTMBHeader.NCFEBs(), theTMBHeader.NTBins(), buf+e0bLine+1);

  if(!theCLCTData.check())   {
    edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: Bad CLCT data";
  }
  else {
    currentPosition+=theCLCTData.sizeInWords();
  }

  //int i = currentPosition-1;
  //printf ( "%04x %04x %04x %04x\n",buf[i+3],buf[i+2],buf[i+1],buf[i] ) ;
 

  // look for RPC
  int b04Line = currentPosition;
  
  if(buf[b04Line]==0x6b04) {
    // we need an e04 line to calculate the size
    int e04Line = findLine(buf, 0x6e04, currentPosition, currentPosition+MaxSizeRPC);
    if(e04Line != -1) {
      theRPCDataIsPresent = true;
      theRPCData = CSCRPCData(buf+b04Line, e04Line-b04Line+1);
      currentPosition+=theRPCData.sizeInWords();
    }
    else {
      edm::LogError("CSCTMBData") << "CSCTMBData::corrupt RPC data! Failed to find end! ";
      return 0;
    }
  }
  

  int TotTMBReadout=0;
  switch (firmwareVersion) {
  case 2007:
    TotTMBReadout= 43+Ntbins*6*5+1+Ntbins*2*4+2+8*256+8;
    break;
  case 2006:
    TotTMBReadout= 27+Ntbins*6*5+1+Ntbins*2*4+2+8*256+8; //see tmb2004 manual (version v2p06) page54.
    break;
  default:
    edm::LogError("CSCTMBData") << "can't find TotTMBReadout - unknown firmware version!";
    break;
  }

  if (buf[currentPosition]==0x6b05) {
    int b05Line = currentPosition;
    edm::LogInfo("CSCTMBData") <<"found scope!!!!!!!!!!!!!";
    int e05Line = findLine(buf, 0x6e05, currentPosition, TotTMBReadout-currentPosition);
    if(e05Line != -1){     
      theTMBScopeIsPresent = true;
      theTMBScope = new CSCTMBScope(buf,b05Line, e05Line);
      currentPosition+=theTMBScope->sizeInWords();
    }
  }
  
  int maxLine = findLine(buf, 0xde0f, currentPosition, TotTMBReadout-currentPosition);
  if(maxLine == -1) 
    {
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
  int e0cLine = findLine(buf, 0x6e0c, currentPosition, maxLine);
  if (e0cLine == -1)
    {
      edm::LogError("CSCTMBData") << "+++ CSCTMBData warning: No e0c line!";
    } 
  else 
    {
      theTMBTrailer = CSCTMBTrailer(buf+e0cLine, firmwareVersion);
    }

  checkSize();

  // size, since we count from 0 and have one more trailer word
  // there are sometimes multiple "de0f" lines in trailer, so key on "6e0c"
  return e0cLine-b0cLine+theTMBTrailer.sizeInWords();
} //UnpackTMB

bool CSCTMBData::checkSize() const 
{
  // sum up all the components and see if they have the size indicated in the TMBTrailer
  return true;
}

std::bitset<22> CSCTMBData::calCRC22(const std::vector< std::bitset<16> >& datain)
{
  std::bitset<22> CRC;
  CRC.reset();
  for(unsigned int i=0;i<datain.size()-3;++i)
    {
      CRC=nextCRC22_D16(datain[i],CRC);
    }
  return CRC;
}

CSCTMBScope & CSCTMBData::tmbScope() const 
{
  if (!theTMBScopeIsPresent) throw("No TMBScope in this chamber");
  return * theTMBScope;
}


std::bitset<22> CSCTMBData::nextCRC22_D16(const std::bitset<16>& D, 
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


boost::dynamic_bitset<> CSCTMBData::pack() 
{
  boost::dynamic_bitset<> result = bitset_utilities::ushortToBitset(theTMBHeader.sizeInWords()*16,
								    theTMBHeader.data());
  boost::dynamic_bitset<> clctData =  bitset_utilities::ushortToBitset(theCLCTData.sizeInWords()*16,
								       theCLCTData.data());
  result = bitset_utilities::append(result,clctData);
  int finalSize = result.size()/16 + theTMBTrailer.sizeInWords(); //size() returns # of bits 
                                                                
  theTMBTrailer.setWordCount(finalSize);
  boost::dynamic_bitset<> tmbTrailer =  bitset_utilities::ushortToBitset( theTMBTrailer.sizeInWords()*16,
									  theTMBTrailer.data());
  result = bitset_utilities::append(result,tmbTrailer);
  return result;
}

