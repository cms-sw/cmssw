/** \file CSCDCCEventData.cc
 *
 *  $Date: 2010/06/11 15:50:28 $
 *  $Revision: 1.31 $
 *  \author A. Tumanov - Rice - But long, long ago...
 *
 */

#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cstdio>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"

#ifdef LOCAL_UNPACK
bool CSCDCCEventData::debug = false;
#else
#include <atomic>
std::atomic<bool> CSCDCCEventData::debug{false};
#endif

CSCDCCEventData::CSCDCCEventData(int sourceId, int nDDUs, int bx, int l1a) 
: theDCCHeader(bx, l1a, sourceId) 
{
  theDDUData.reserve(nDDUs);
} 

CSCDCCEventData::CSCDCCEventData(const uint16_t *buf, CSCDCCExaminer* examiner)
{
  unpack_data(buf, examiner);
}

CSCDCCEventData::~CSCDCCEventData() 
{
}


void CSCDCCEventData::unpack_data(const uint16_t *buf, CSCDCCExaminer* examiner) 
{
 
/*
  for (int i=0;i<20;i++) {
    printf("%04x %04x %04x %04x\n",buf[i+3],buf[i+2],buf[i+1],buf[i]); 
    i+=3;
  }
*/

  theDDUData.clear();
  if (debug) 
    LogTrace ("CSCDCCEventData|CSCRawToDigi") << "CSCDCCEventData::unpack_data() is called";

  // decode DCC header (128 bits)
  if (debug) 
    LogTrace ("CSCDCCEventData|CSCRawToDigi") << "unpacking dcc header...";
  theDCCHeader.setFromBuffer(buf);
  //theDCCHeader = CSCDCCHeader(buf); // direct unpacking instead of bitfields
  buf += theDCCHeader.sizeInWords();

  //std::cout <<"Sandrik DCC Id = " << theDCCHeader.getCDFSourceId()  << std::endl;
 
  ///loop over DDUEventDatas
  while ( (buf[7]==0x8000)&&(buf[6]==0x0001)&&(buf[5]==0x8000))
    {
       CSCDDUEventData dduEventData(buf, examiner);
//	CSCDDUEventData dduEventData(buf);

      if (debug) LogTrace ("CSCDCCEventData|CSCRawToDigi") << " checking ddu data integrity ";
      if (dduEventData.check()) 
	{
	  theDDUData.push_back(dduEventData);
	  buf += dduEventData.sizeInWords();
	} 
      else
	{
	  if (debug) LogTrace("CSCDCCEventData|CSCRawToDigi") <<"DDU Data Check failed!  ";
	  break;
	}
      
    }
  
  if (debug)
    {
      LogTrace ("CSCDCCEventData|CSCRawToDigi") << "unpacking dcc trailer ";
      LogTrace ("CSCDCCEventData|CSCRawToDigi") << std::hex << buf[3] <<" "
				       << buf[2]<<" " << buf[1]<<" " << buf[0];
    }
	    
  //decode dcc trailer (128 bits)
  if (debug) LogTrace ("CSCDCCEventData|CSCRawToDigi") <<"decoding DCC trailer";
  theDCCTrailer.setFromBuffer(buf);
  if (debug) LogTrace("CSCDCCEventData|CSCRawToDigi") << "checking DDU Trailer" << theDCCTrailer.check(); 
 
  // buf += theDCCTrailer.sizeInWords(); /* =VB= Commented out to please static analyzer */ 

  //std::cout << " DCC Size: " << std::dec << theSizeInWords << std::endl;
  //std::cout << "LastBuf: "  << std::hex << inputBuf[theSizeInWords-4] << std::endl;
}
	  

bool CSCDCCEventData::check() const 
{
  // the trailer counts in 64-bit words
  if (debug) 
    {
      LogTrace ("CSCDCCEventData|CSCRawToDigi") << "size in Words () = " << std::dec << sizeInWords();
    }

  return  theDCCHeader.check() && theDCCTrailer.check();
}


void CSCDCCEventData::addChamber(CSCEventData & chamber, int dduID, int dduSlot, int dduInput, int dmbID, uint16_t format_version)
{
  // first, find this DDU
  std::vector<CSCDDUEventData>::iterator dduItr;
  int dduIndex = -1;
  int nDDUs = theDDUData.size();
  for(int i = 0; dduIndex == -1 && i < nDDUs; ++i)
  {
    if(theDDUData[i].header().source_id() == dduID) dduIndex = i;
  }
  
  /// Set DDU format_version field in header depending on desired format version
  unsigned ddu_fmt_version = 0x6; // 2005 Format
  if (format_version == 2013) ddu_fmt_version = 0x7; /// 2013 Format
  
  if(dduIndex == -1)
  {
    // make a new one
    CSCDDUHeader newDDUHeader(dccHeader().getCDFBunchCounter(), 
                              dccHeader().getCDFEventNumber(), dduID, ddu_fmt_version);
    theDDUData.push_back(CSCDDUEventData(newDDUHeader));
    dduIndex = nDDUs;
    dccHeader().setDAV(dduSlot);
  }
  theDDUData[dduIndex].add( chamber, dmbID, dduInput, format_version);
}
 

boost::dynamic_bitset<> CSCDCCEventData::pack() 
{
  boost::dynamic_bitset<> result( theDCCHeader.sizeInWords()*16);
  result = bitset_utilities::ushortToBitset(theDCCHeader.sizeInWords()*16, theDCCHeader.data());
  //std::cout <<"SANDRIK DCC size of header  in words"<< theDCCHeader.sizeInWords()*16 <<std::endl;  
  //std::cout <<"SANDRIK DCC size of header in bits"<< result.size()<<std::endl;
  //for(size_t i = 0; i < result.size(); ++i) {
  //  std::cout<<result[i];
  //  if (((i+1)%32)==0) std::cout<<std::endl;
  //}
  
  for(size_t i = 0; i < theDDUData.size(); ++i) 
    {
      result = bitset_utilities::append(result,theDDUData[i].pack());
      //std::cout <<"SANDRIK here is ddu data check ";
      //theDDUData[i].header().check();
      //std::cout <<std::endl;
      //bitset_utilities::printWords(result);
    }
  
  //std::cout <<"SANDRIK packed dcc size is "<<result.size()<<std::endl;
  //for(size_t i = 0; i < result.size(); ++i) {
  //  std::cout<<result[i];
  //  if (((i+1)%32)==0) std::cout<<std::endl;
  //}

  boost::dynamic_bitset<> dccTrailer = bitset_utilities::ushortToBitset(theDCCTrailer.sizeInWords()*16,
									theDCCTrailer.data());
  result = bitset_utilities::append(result,dccTrailer);
  //  bitset_utilities::printWords(result);
  return result;
}

