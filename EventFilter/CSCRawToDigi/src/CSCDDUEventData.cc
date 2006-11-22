/* CSCDDUEventData.cc
 * Modified 4/21/03 to get rid of arrays and store all CSC data 
 * in vectors. 
 * A.Tumanov
 */

#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCTrailer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>
#include <cstdio>

#include <boost/dynamic_bitset.hpp>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"

bool CSCDDUEventData::debug = false;
unsigned int CSCDDUEventData::errMask = 0xFFFFFFFF;


CSCDDUEventData::CSCDDUEventData(const CSCDDUHeader & header) {
  theDDUHeader = header;
}

  
CSCDDUEventData::CSCDDUEventData(unsigned short *buf) {
  unpack_data(buf);
}

CSCDDUEventData::~CSCDDUEventData() {
}


void CSCDDUEventData::add(CSCEventData & cscData) {
  theDDUHeader.ncsc_++;
  cscData.setEventInformation(theDDUHeader.bxnum(), theDDUHeader.lvl1num());
  theData.push_back(cscData);
}

void CSCDDUEventData::decodeStatus() const {
  this->decodeStatus(theDDUTrailer.errorstat());
}

void CSCDDUEventData::decodeStatus(int code) const {
  //JRG is Jason Gilmore
  // JRG, low-order 16-bit status (most serious errors):
  if((code&errMask)>0){///this is a mask for printing out errors
    // JRG, low-order 16-bit status (most serious errors):
    if((code&0x0000F000)>0){
      if((0x00008000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Critical Error, ** needs reset **";
      if((0x00004000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Single Error, bad event";
      if((0x00002000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Single Warning";
      if((0x00001000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Near Full Warning";
	}
    if((code&0x00000F00)>0){
      if((0x00000800&code)>0) 
	edm::LogError ("CSCDDUEventData") << "   DDU 64-bit Alignment Error";
      if((0x00000400&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Control DLL Error occured";
      if((0x00000200&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU DMB Error occurred";
      if((0x00000100&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Lost In Event Error";
    }
    
    if((code&0x000000F0)>0){
      if((0x00000080&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Lost In Data Error occurred";
      if((0x00000040&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Timeout Error";
      if((0x00000020&code)>0)
	edm::LogError ("CSCDDUEventData") << "   TMB or ALCT CRC Error";
      if((0x00000010&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Multiple Transmit Errors";
    }
    if((code&0x0000000F)>0){
      if((0x00000008&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Sync Lost or FIFO Full Error";
      if((0x00000004&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Fiber/FIFO Connection Error";
      if((0x00000002&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU L1A Match Error";
      if((0x00000001&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU DMB or CFEB CRC Error";
    }
    if((code&0xF0000000)>0){
      // JRG, high-order 16-bit status (not-so-serious errors):
      if((0x80000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU DMB LCT/DAV/Movlp Mismatch";
      if((0x40000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU-CFEB L1 Mismatch";
      if((0x20000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU saw no good DMB CRCs";
      if((0x10000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU CFEB Count Error";
     
    }
    if((code&0x0F000000)>0){
      if((0x08000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU FirstDat Error";
      if((0x04000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU L1A-FIFO Full Error";
      if((0x02000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Data Stuck in FIFO";
      if((0x01000000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU NoLiveFibers Error";
    }
    if((code&0x00F00000)>0){
      if((0x00800000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Spwd single-bit Warning";
      if((0x00400000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Input FPGA Error";
      if((0x00200000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU DAQ Stop bit set";
      if((0x00100000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU DAQ says Not Ready";
      if((0x00300000&code)==0x00200000)
        edm::LogError ("CSCDDUEventData") << "   DDU DAQ Applied Backpressure";
    }

    if((code&0x000F0000)>0){
      if((0x00080000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU TMB Error";
      if((0x00040000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU ALCT Error";
      if((0x00020000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Trigger Readout Wordcount Error";
      if((0x00010000&code)>0)
	edm::LogError ("CSCDDUEventData") << "   DDU Trigger L1A Match Error";
    }
  }
}

void CSCDDUEventData::unpack_data(unsigned short *buf) {
  // just to calculate length
  unsigned short * inputBuf = buf;
  theData.clear();
  if (debug) edm::LogInfo ("CSCDDUEventData") << "CSCDDUEventData::unpack_data() is called";

  if (debug) for (int i=0;i<4;i++) {
    edm::LogInfo ("CSCDDUEventData") << i << std::hex << buf[4*i+3] << buf[4*i+2] 
				     << buf[4*i+1] << buf[4*i];
  }

  memcpy(&theDDUHeader, buf, theDDUHeader.sizeInWords()*2);

  if (debug) {
    edm::LogInfo ("CSCDDUEventData") << "size of ddu header in words = " << theDDUHeader.sizeInWords();
    edm::LogInfo ("CSCDDUEventData") << "sizeof(DDUHeader) = " << sizeof(theDDUHeader);
  }
  buf += theDDUHeader.sizeInWords();

  std::cout << "sandrik dduID =" << theDDUHeader.source_id() << std::endl; 
  
  int i=-1;
 
  // we really don't want to copy CSCEventData's while filling the vec
  theData.clear();
  theData.reserve(theDDUHeader.ncsc());

  while( (((buf[0]&0xf000) == 0x9000)||((buf[0]&0xf000) == 0xa000)) 
          && (buf[3] != 0x8000)){
    ++i;
    if (debug) edm::LogInfo ("CSCDDUEventData") << "unpack csc data loop started";
    theData.push_back(CSCEventData(buf));
    buf += (theData.back()).size();
    if (debug) {
      edm::LogInfo ("CSCDDUEventData") << "size of vector of cscDatas = " << theData.size();
    }
  }


  if (debug) {
    edm::LogInfo ("CSCDDUEventData") << "unpacking ddu trailer ";
    edm::LogInfo ("CSCDDUEventData") << std::hex << buf[3]<<" " << buf[2] 
				     <<" " << buf[1]<<" " << buf[0];
  }

  // decode ddu tail
  memcpy(&theDDUTrailer, buf, theDDUTrailer.sizeInWords()*2);
  if (debug) edm::LogInfo ("CSCDDUEventData") << theDDUTrailer.check();
  errorstat=theDDUTrailer.errorstat();
  if (errorstat&errMask != 0)  {
    if (theDDUTrailer.check()) {
      edm::LogError ("CSCDDUEventData") 
	<< "+++ CSCDDUEventData warning: DDU Trailer errors = " << std::hex << errorstat << " +++ ";
      decodeStatus(errorstat);
    } else {
      edm::LogError ("CSCDDUEventData" ) 
	<< " Unpacking lost DDU trailer - check() failed and 8 8 ffff 8 was not found ";
    }
  }
   
  if (debug) 
    edm::LogInfo ("CSCDDUEventData")  << " Final errorstat " << std::hex << errorstat << std::dec ;
  // the trailer counts in 64-bit words
  buf += theDDUTrailer.sizeInWords();
  
  theSizeInWords = buf - inputBuf;
}


bool CSCDDUEventData::check() const {
  // the trailer counts in 64-bit words
  if (debug) {
    edm::LogInfo ("CSCDDUEventData") << sizeInWords();
    edm::LogInfo ("CSCDDUEventData") << "wordcount = " << theDDUTrailer.wordcount()*4;
  }

  return theDDUHeader.check() && theDDUTrailer.check();
}

boost::dynamic_bitset<> CSCDDUEventData::pack() {
  
  boost::dynamic_bitset<> result = bitset_utilities::ushortToBitset( theDDUHeader.sizeInWords()*16,
								     theDDUHeader.data());
 
    
  for(unsigned int i = 0; i < theData.size(); ++i) {
    result = bitset_utilities::append(result,theData[i].pack());
  }
  
  boost::dynamic_bitset<> dduTrailer = bitset_utilities::ushortToBitset ( theDDUTrailer.sizeInWords()*16, 
									  theDDUTrailer.data());
  result =  bitset_utilities::append(result,dduTrailer);

  return result;
}

