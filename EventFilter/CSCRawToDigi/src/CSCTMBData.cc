/** \class CSCTMBData
 *
 *  $Date: 2012/01/10 17:06:31 $
 *  $Revision: 1.32 $
 *  \author A. Tumanov - Rice
 */

#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>  // dump for JK
#include <iostream>
#include <cstdio>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"
#include "EventFilter/CSCRawToDigi/src/cscPackerCompare.h"

bool CSCTMBData::debug =false;

CSCTMBData::CSCTMBData() 
  : theOriginalBuffer(0), 
    theB0CLine( 0 ),
    theE0FLine( 0 ),
    theTMBHeader(2007, 0x50c3),
    theCLCTData(&theTMBHeader),
    theTMBScopeIsPresent(false), 
    theTMBScope(0),
    theTMBMiniScopeIsPresent(false), 
    theTMBMiniScope(0),
    theBlockedCFEBIsPresent(false),
    theTMBBlockedCFEB(0),
    theTMBTrailer(theTMBHeader.sizeInWords()+theCLCTData.sizeInWords(), 2007),
    size_( 0 ), 
    cWordCnt( 0 ),
    theRPCDataIsPresent(false)
{


}


CSCTMBData::CSCTMBData(unsigned short *buf) 
  : theOriginalBuffer(buf), 
    theTMBHeader(2007, 0x50c3),
    theCLCTData(&theTMBHeader),
    theTMBScopeIsPresent(false), 
    theTMBScope(0), 
    theTMBMiniScopeIsPresent(false), 
    theTMBMiniScope(0), 
    theBlockedCFEBIsPresent(false),
    theTMBBlockedCFEB(0),
    theTMBTrailer(theTMBHeader.sizeInWords()+theCLCTData.sizeInWords(), 2007),
    theRPCDataIsPresent(false){
  size_ = UnpackTMB(buf);
} 

// Explicitly-defined copy constructor is needed when the scope data is
// present, to prevent the same pointer from being deleted twice. -SV.
CSCTMBData::CSCTMBData(const CSCTMBData& data):
  theOriginalBuffer(data.theOriginalBuffer),
  theB0CLine(data.theB0CLine), theE0FLine(data.theE0FLine),
  theTMBHeader(data.theTMBHeader),
  theCLCTData(data.theCLCTData), theRPCData(data.theRPCData),
  theTMBScopeIsPresent(data.theTMBScopeIsPresent),
  theTMBMiniScopeIsPresent(data.theTMBMiniScopeIsPresent),
  theBlockedCFEBIsPresent(data.theBlockedCFEBIsPresent),
  theTMBTrailer(data.theTMBTrailer),
  size_(data.size_), cWordCnt(data.cWordCnt),
  theRPCDataIsPresent(data.theRPCDataIsPresent)
{
  if (theTMBScopeIsPresent) {
    theTMBScope = new CSCTMBScope(*(data.theTMBScope));
  }
  else {
    theTMBScope = 0;
  }
  
  if (theTMBMiniScopeIsPresent) {
    theTMBMiniScope = new CSCTMBMiniScope(*(data.theTMBMiniScope));
  }
  else {
    theTMBMiniScope = 0;
  }
  
  if (theBlockedCFEBIsPresent) {
     theTMBBlockedCFEB = new CSCTMBBlockedCFEB(*(data.theTMBBlockedCFEB));
  }
  else {
    theTMBBlockedCFEB = 0;
  }
  
}

CSCTMBData::~CSCTMBData(){
  if (theTMBScopeIsPresent) {
    delete theTMBScope;
    theTMBScopeIsPresent = false;
  }

  if (theTMBMiniScopeIsPresent) {
    delete theTMBMiniScope;
    theTMBMiniScopeIsPresent = false;
  }

  if (theBlockedCFEBIsPresent) {
    delete theTMBBlockedCFEB;
    theBlockedCFEBIsPresent = false;
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
    LogTrace("CSCTMBData|CSCRawToDigi") << " Test here " << CRC.to_ulong();
    return CRC.to_ulong();
  } 
  else {
    LogTrace("CSCTMBData|CSCRawToDigi") << "theTotalTMBData doesn't exist";
    return 0;
  }
}

int CSCTMBData::UnpackTMB(unsigned short *buf) {
  ///determine 2007 or 2006 version
  unsigned short int firmwareVersion=0;
  int Ntbins = 0 ;
  int NHeaderFrames = 0; //WARNING in 5_0_X
  int NRPCtbins = 0; // =VB= number of RPC tbins  
  
  int b0cLine=0;///assumes that buf starts at the tmb data
                ///this is not true if something is wrong in the data 
                ///before TMB - then we skip the whole event

  NHeaderFrames++; NHeaderFrames--;

  if (buf[b0cLine]==0xdb0c) {
    firmwareVersion=2007;
    Ntbins = buf[b0cLine+19]&0xF8;
    NRPCtbins = (buf[b0cLine+36]>>5)&0x1F; // =VB= get RPC tbins  
    NHeaderFrames = buf[b0cLine+5]&0x3F; //WARNING in 5_0_X
  } 
  else if (buf[b0cLine]==0x6b0c) {
    firmwareVersion=2006;
    Ntbins =  buf[b0cLine+1]&0x1f ;
    NRPCtbins = Ntbins;
    NHeaderFrames = buf[b0cLine+4]&0x1f; //WARNING in 5_0_X
  } 
  else {
    LogTrace("CSCTMBData|CSCRawToDigi") << "+++ Can't find b0C flag";
  }

  if ((firmwareVersion==2007)&&(!(((buf[b0cLine]&0xFFFF)==0xDB0C)&&((buf[b0cLine+1]&0xf000)==0xD000)
	&&((buf[b0cLine+2]&0xf000)==0xD000)&&((buf[b0cLine+3]&0xf000)==0xD000)))){
    LogTrace("CSCTMBData|CSCRawToDigi") << "+++ CSCTMBData warning: error in header in 2007 format!";
  }

  int MaxSizeRPC = 1+NRPCtbins*2*4+1;
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
    edm::LogError("CSCTMBData|CSCRawToDigi") << "+++ undetermined firmware format - cant find e0bLine";
  }

  theTMBHeader=CSCTMBHeader(buf);
  
  if(!theTMBHeader.check())   {
    LogTrace("CSCTMBData|CSCRawToDigi") << "+++ CSCTMBData warning: Bad TMB header e0bLine=" << std::hex << buf[e0bLine];
    return 0;
  }

  int currentPosition = theTMBHeader.sizeInWords();

  theCLCTData = CSCCLCTData(theTMBHeader.NCFEBs(), theTMBHeader.NTBins(), buf+e0bLine+1);

  if(!theCLCTData.check())   {
    LogTrace("CSCTMBData|CSCRawToDigi") << "+++ CSCTMBData warning: Bad CLCT data";
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
      LogTrace("CSCTMBData|CSCRawToDigi") << "CSCTMBData::corrupt RPC data! Failed to find end! ";
      return 0;
    }
  }

  int TotTMBReadout=0;
  switch (firmwareVersion) {
  case 2007:
    TotTMBReadout= 43+Ntbins*6*5+1+NRPCtbins*2*4+2+8*256+8;
    break;
  case 2006:
    TotTMBReadout= 27+Ntbins*6*5+1+NRPCtbins*2*4+2+8*256+8; //see tmb2004 manual (version v2p06) page54.
    break;
  default:
    edm::LogError("CSCTMBData|CSCRawToDigi") << "can't find TotTMBReadout - unknown firmware version!";
    break;
  }
  
//std::cout << " !!!TMB Scope!!! " << std::endl;
  if (buf[currentPosition]==0x6b05) {
    int b05Line = currentPosition;
    LogTrace("CSCTMBData|CSCRawToDigi") << "found scope!";
    int e05Line = findLine(buf, 0x6e05, currentPosition, TotTMBReadout-currentPosition);
    if(e05Line != -1){
      theTMBScopeIsPresent = true;
      theTMBScope = new CSCTMBScope(buf,b05Line, e05Line);
      // The size of the TMB scope data can vary, and I see no good reasons
      // not to determine it dynamically.  -SV, 5 Nov 2008.
      //currentPosition+=theTMBScope->sizeInWords();
      currentPosition+=(e05Line-b05Line+1);
    }
    else {
      LogTrace("CSCTMBData|CSCRawToDigi")
	<< "+++ CSCTMBData warning: found 0x6b05 line, but not 0x6e05! +++";
    }
  }

  /// Now Find the miniscope
  if (buf[currentPosition]==0x6b07){
     int Line6b07 = currentPosition;
     LogTrace("CSCTMBData") << " TMBData ---> Begin of MiniScope found " ;
     int Line6E07 = findLine(buf, 0x6E07, currentPosition, TotTMBReadout-currentPosition);
     if(Line6E07 !=-1){
       LogTrace("CSCTMBData") << " TMBData --> End of MiniScope found " << Line6E07-Line6b07+1 << " words ";
       theTMBMiniScopeIsPresent = true;
       theTMBMiniScope = new CSCTMBMiniScope(buf, Line6b07, Line6E07);
       currentPosition += (Line6E07-Line6b07+1);
     }
     else {
      LogTrace("CSCTMBData")
	<< "+++ CSCTMBData warning MiniScope!: found 0x6b07 line, but not 0x6e07! +++";
    }
  }
  /// end for the mini scope

 /// Now Find the blocked CFEB DiStrips List Format
  if (buf[currentPosition]==0x6BCB){
  int Line6BCB = currentPosition;
     LogTrace("CSCTMBData") << " TMBData ---> Begin of Blocked CFEB found " ;
     int Line6ECB = findLine(buf, 0x6ECB, currentPosition, TotTMBReadout-currentPosition);
     if(Line6ECB !=-1){
       LogTrace("CSCTMBData") << " TMBData --> End of Blocked CFEB found " << Line6ECB-Line6BCB+1 << " words ";
       theBlockedCFEBIsPresent = true;
       theTMBBlockedCFEB = new CSCTMBBlockedCFEB(buf, Line6BCB, Line6ECB);
       currentPosition += (Line6ECB-Line6BCB+1);
     }
     else {
      LogTrace("CSCTMBData")
	<< "+++ CSCTMBData warning Blocked CFEB!: found 0x6BCB line, but not 0x6ECB! +++";
    }
  }
 /// end for the blocked CFEB DiStrips List Format  

  int maxLine = findLine(buf, 0xde0f, currentPosition, TotTMBReadout-currentPosition);
  if(maxLine == -1) 
    {
      LogTrace("CSCTMBData|CSCRawToDigi") << "+++ CSCTMBData warning: No e0f line!";
      return 0;
    }

  //Now for CRC check put this information into bitset

  theB0CLine = b0cLine;
  theE0FLine = maxLine;

  // finally, the trailer
  int e0cLine = findLine(buf, 0x6e0c, currentPosition, maxLine);
  if (e0cLine == -1)
    {
      LogTrace("CSCTMBData|CSCRawToDigi") << "+++ CSCTMBData warning: No e0c line!";
    } 
  else 
    {
      theTMBTrailer = CSCTMBTrailer(buf+e0cLine, firmwareVersion);
      LogTrace("CSCTMBData|CSCRawToDigi")
	<< "TMB trailer size: " << theTMBTrailer.sizeInWords();
    }

  checkSize();

  // Dump of TMB; format proposed by JK.
#ifdef TMBDUMP
  LogTrace("CSCTMBData") << "Dump of TMB data:";
  for (int line = b0cLine; line <= maxLine+3; line++) {
    LogTrace("CSCTMBData")
      << "Adr= " << std::setw(4) << line
      << " Data= " << std::setfill('0') << std::setw(5)
      << std::uppercase << std::hex << buf[line] << std::dec << std::endl;
  }
#endif

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

CSCTMBMiniScope & CSCTMBData::tmbMiniScope() const
{
  if (!theTMBMiniScopeIsPresent) throw("No TMBScope in this chamber");
  return * theTMBMiniScope;
}


CSCTMBBlockedCFEB & CSCTMBData::tmbBlockedCFEB() const
{
  if (!theBlockedCFEBIsPresent) throw("No TMB Blocked CFEB in this chamber");
  return * theTMBBlockedCFEB;
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
  boost::dynamic_bitset<> newResult = result;
//  theTMBTrailer.setCRC(TMBCRCcalc());

  boost::dynamic_bitset<> tmbTrailer =  bitset_utilities::ushortToBitset( theTMBTrailer.sizeInWords()*16,
									  theTMBTrailer.data());
  result = bitset_utilities::append(result,tmbTrailer);
  
  // now convert to a vector<bitset<16>>, so we can calculate the crc
  std::vector<std::bitset<16> > wordVector;
  // try to tune so it stops before the e0f line
  for(unsigned pos = 0; pos < result.size()-16; pos += 16)
  {
    std::bitset<16> word;
    for(int i = 0; i < 16; ++i)
    {
      word[i] = result[pos+i];
    }
    wordVector.push_back(word);
  }
  theTMBTrailer.setCRC(calCRC22(wordVector).to_ulong());
  tmbTrailer =  bitset_utilities::ushortToBitset( theTMBTrailer.sizeInWords()*16,
                                                  theTMBTrailer.data());
  newResult = bitset_utilities::append(newResult, tmbTrailer);

  return newResult;
}


void CSCTMBData::selfTest()
{
  CSCTMBData tmbData;
  cscClassPackerCompare(tmbData);
}

