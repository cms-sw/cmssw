/// CSCDCCEventData.cc
/// 01/20/05 
/// A.Tumanov

#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <vector>
#include <cstdio>
#include <boost/dynamic_bitset.hpp>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"

bool CSCDCCEventData::debug = false;

CSCDCCEventData::CSCDCCEventData(int sourceId, int nDDUs, int bx, int l1a) 
: theDCCHeader(bx, l1a, sourceId) 
{

  theDDUData.reserve(nDDUs);
  CSCDDUHeader dduHeader(bx, l1a, sourceId);
  for(int i = 0; i < nDDUs; ++i) {
    theDDUData.push_back(CSCDDUEventData(dduHeader));
  }
} 
  
CSCDCCEventData::CSCDCCEventData(unsigned short *buf) {
  unpack_data(buf);
}

CSCDCCEventData::~CSCDCCEventData() {
}


void CSCDCCEventData::unpack_data(unsigned short *buf) {
 
  theDDUData.clear();
  if (debug) 
    edm::LogInfo ("CSCDCCEventData") << "CSCDCCEventData::unpack_data() is called";

  // decode DCC header (128 bits)
  if (debug) 
    edm::LogInfo ("CSCDCCEventData") << "unpacking dcc header...";
  memcpy(&theDCCHeader, buf, theDCCHeader.sizeInWords()*2);
  //theDCCHeader = CSCDCCHeader(buf); // direct unpacking instead of bitfields
  buf += theDCCHeader.sizeInWords();

  std::cout <<"Sandrik DCC Id = " << theDCCHeader.getCDFSourceId()  << std::endl;
 
  ///loop over DDUEventDatas
  while ( (buf[7]==0x8000)&&(buf[6]==0x0001)&&(buf[5]==0x8000))
  {
    CSCDDUEventData dduEventData(buf);
    if (debug) edm::LogInfo ("CSCDCCEventData") << " checking ddu data integrity ";
    if (dduEventData.check()) {
      theDDUData.push_back(dduEventData);
      buf += dduEventData.sizeInWords();
    } else {
      edm::LogError ("CSCDCCEventData") <<"DDU Data Check failed! reasons:  "
					<< "size of dduData= " << dduEventData.size() 
					<< "sizeof( dduData) =  " << sizeof(dduEventData);
      break;
    }
    
  }
  


  if (debug) {
    edm::LogInfo ("CSCDCCEventData") << "unpacking dcc trailer ";
    edm::LogInfo ("CSCDCCEventData") << std::hex << buf[3] <<" "
				     << buf[2]<<" " << buf[1]<<" " << buf[0];
  }
	    
  //decode dcc trailer (128 bits)
  if (debug) edm::LogInfo ("CSCDCCEventData") <<"decoding DCC trailer";
  memcpy(&theDCCTrailer, buf, theDCCTrailer.sizeInWords()*2);
  if (debug) edm::LogInfo ("CSCDCCEventData") << "checking DDU Trailer" << theDCCTrailer.check(); 
  buf += theDCCTrailer.sizeInWords();

}
	  
bool CSCDCCEventData::check() const {
  // the trailer counts in 64-bit words
  if (debug) {
    edm::LogInfo ("CSCDCCEventData") << "size in Words () = " << std::dec << sizeInWords();
  }

  return  theDCCHeader.check() && theDCCTrailer.check();
}

boost::dynamic_bitset<> CSCDCCEventData::pack() {

  boost::dynamic_bitset<> result; 
  boost::dynamic_bitset<> dccHeader( theDCCHeader.sizeInWords()*16, *(const unsigned int *)&theDCCHeader);
  result = dccHeader;

  for(size_t i = 0; i < theDDUData.size(); ++i) {
    result = bitset_utilities::append(result,theDDUData[i].pack());
  }
  boost::dynamic_bitset<> dccTrailer( theDCCTrailer.sizeInWords()*16, *(const unsigned *)&theDCCTrailer);
  result = bitset_utilities::append(result,dccTrailer);

  return result;
}

