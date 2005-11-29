/// CSCDCCEventData.cc
/// 01/20/05 
/// A.Tumanov


#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"

//#include "Utilities/GenUtil/interface/BitVector.h"
#include <iostream>
#include <vector>
#include <cstdio>

bool CSCDCCEventData::debug = true;

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
  if (debug) std::cout << "CSCDCCEventData::unpack_data() is called" << std::endl;

  // decode DCC header (128 bits)
  if (debug) std::cout << "unpacking dcc header..." << std::endl;
  memcpy(&theDCCHeader, buf, theDCCHeader.sizeInWords()*2);
  //theDCCHeader = CSCDCCHeader(buf); // direct unpacking instead of bitfields
  buf += theDCCHeader.sizeInWords();

  if (debug) for (int i=0;i<4;i++) {
    printf("%d   %04x %04x %04x %04x\n",i,buf[4*i+3],buf[4*i+2],buf[4*i+1],buf[4*i]);
  }

 
  ///loop over DDUEventDatas
  while ( (buf[7]==0x8000)&&(buf[6]==0x0001)&&(buf[5]==0x8000))
  {
    if (debug) std::cout << std::endl << "unpack ddu data loop started" << std::endl;
    CSCDDUEventData dduEventData(buf);
    if (debug)  std::cout << " checking ddu data integrity "<< std::endl;
    if (dduEventData.check()) {
      theDDUData.push_back(dduEventData);
      buf += dduEventData.size();
      if (debug) {
	std::cout << "size of vector of dduDatas = " << theDDUData.size() << std::endl;
	printf("%04x %04x %04x %04x\n",buf[3],buf[2],buf[1],buf[0]);
      }
    } else {
      std::cout <<"DDU Data Check failed! reasons:  ";
      std::cout << "size of dduData= " << dduEventData.size() << std::endl;
      std::cout << "sizeof( dduData) =  " << sizeof(dduEventData) << std::endl;

      for (int i=0;i<20;i++) {
	printf("%04x %04x %04x %04x\n",buf[i+3],buf[i+2],buf[i+1],buf[i]);
      }
    }
    
  }
  


  if (debug) {
    std::cout << "unpacking dcc trailer " << std::endl;
    printf("%04x %04x %04x %04x\n",buf[3],buf[2],buf[1],buf[0]);
  }
	    
  //decode dcc trailer (128 bits)
  if (debug)  std::cout<<"decoding DCC trailer" << std::endl;
  memcpy(&theDCCTrailer, buf, theDCCTrailer.sizeInWords()*2);
  if (debug) std::cout << "checking DCC trailer " << theDCCTrailer.check() << std::endl; 
  buf += theDCCTrailer.sizeInWords();

}
	  
bool CSCDCCEventData::check() const {
  // the trailer counts in 64-bit words
  if (debug) {
    std::cout << "size in Words () = " << std::dec << sizeInWords() << std::endl;
    //std::cout << "wordcount = " << std::dec << theDDUTrailer.wordcount()*4 << std::hex<< std::endl;
    //std::cout << "DDUHeader check " << std::hex <<theDDUHeader.check() << std::endl;
    //std::cout << "DDUTrailer.check " << theDDUTrailer.check() << std::endl;
  }

  return //(sizeInWords() == theDDUTrailer.word_count * 4) &&
          theDCCHeader.check() && theDCCTrailer.check();
}


/* BitVector CSCDCCEventData::pack() {
  BitVector result;

  Headers should be added by Daq package
  BitVector dccHeader((const unsigned *) &theDCCHeader,
                      theDCCHeader.sizeInWords()*16);
  result.assign(result.nBits(), dccHeader.nBits(), dccHeader);

  for(size_t i = 0; i < theDDUData.size(); ++i) {
    BitVector dduData = theDDUData[i].pack();
    result.assign(result.nBits(), dduData.nBits(), dduData);
  }

 As well as trailers
  BitVector dccTrailer((const unsigned *) &theDCCTrailer,
                      theDCCTrailer.sizeInWords()*16);
  result.assign(result.nBits(), dccTrailer.nBits(), dccTrailer);


  return result;
}
*/
