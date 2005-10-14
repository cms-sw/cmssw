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

//#include "Utilities/GenUtil/interface/BitVector.h"

#include <iostream>
#include <vector>
#include <cstdio>

bool CSCDDUEventData::debug = false;

bool CSCDDUEventData::dccData = false;


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
  // JRG, low-order 16-bit status (most serious errors):
  if((code&0x0000F000)>0){
    if((0x00008000&code)>0)printf("   DDU Critical Error, ** needs reset **\n");
    if((0x00004000&code)>0)printf("   DDU Single Error, bad event\n");
    // JRG, low-order 16-bit status (most serious errors):
    if((code&0x0000F000)>0){
      if((0x00008000&code)>0)printf("   DDU Critical Error, ** needs reset **\n");
      if((0x00004000&code)>0)printf("   DDU Single Error, bad event\n");
      if((0x00002000&code)>0)printf("   DDU Single Warning\n");
      if((0x00001000&code)>0)printf("   DDU Near Full Warning\n");
      printf("\n");
    }
    if((code&0x00000F00)>0){
      if((0x00000800&code)>0)printf("   DDU RX Error\n");
      if((0x00000400&code)>0)printf("   DDU Control DLL Error occured\n");
      if((0x00000200&code)>0)printf("   DDU DMB Error occurred\n");
      if((0x00000100&code)>0)printf("   DDU Lost In Event Error\n");
      printf("\n");
    }
    if((code&0x000000F0)>0){
      if((0x00000080&code)>0)printf("   DDU Lost In Data Error occurred\n");
      if((0x00000040&code)>0)printf("   DDU Timeout Error\n");
 
      // Multiple-bit vote failures (or Rx Errors) in one 64-bit word:
      if((0x00000020&code)>0)printf("   DDU Critical Data Error For OLD DDU\n");
      // Multiple-bit vote failures (or Rx Errors) in one 64-bit word:
      if((0x00000020&code)>0)printf("   TMB or ALCT CRC Error For NEW DDU\n");


      // Multiple single-bit vote failures (or Rx Errors) over time from one DMB:
      if((0x00000010&code)>0)printf("   DDU Multiple Transmit Errors\n");
      printf("\n");
    }
    if((code&0x0000000F)>0){
      if((0x00000008&code)>0)printf("   DDU FIFO Full Error\n");
      if((0x00000004&code)>0)printf("   DDU Fiber Error\n");
      if((0x00000002&code)>0)printf("   DDU L1A Match Error\n");
      if((0x00000001&code)>0)printf("   DDU CRC Error\n");
      printf("\n");
    }
    if((code&0xF0000000)>0){
      // JRG, high-order 16-bit status (not-so-serious errors):
      if((0x80000000&code)>0)printf("   DDU Output Limited Buffer Overflow\n");
      if((0x40000000&code)>0)printf("   DDU G-Bit FIFO Full Warning\n");
      //pre-ddu3ctrl_v12:   if((0x20000000&code)>0)printf("   DDU G-Bit FIFO Near Full Warning");
      if((0x20000000&code)>0)printf("   DDU Ethernet Xmit Limit flag\n");
      if((0x10000000&code)>0)printf("   DDU G-Bit Fiber Error\n");
      printf("\n");
    }
    if((code&0x0F000000)>0){
      if((0x08000000&code)>0)printf("   DDU FirstDat Error\n");
      //Pre-ddu3ctrl_v8r15576:   if((0x04000000&code)>0)printf("   DDU BadFirstWord Error");
      if((0x04000000&code)>0)printf("   DDU L1A-FIFO Full Error\n");
      //Pre-ddu2ctrl_ver53:   if((0x02000000&code)>0)printf("   DDU BadCtrlWord Error");
      if((0x02000000&code)>0)printf("   DDU Data Stuck in FIFO\n");
      if((0x01000000&code)>0)printf("   DDU NoLiveFibers Error\n");
      printf("\n");
    }
    if((code&0x00F00000)>0){
      if((0x00800000&code)>0)printf("   DDU Spwd single-bit Warning\n");
      if((0x00400000&code)>0)printf("   DDU Ethernet DLL Error\n");
      if((0x00200000&code)>0)printf("   DDU S-Link Full Bit set\n");
      if((0x00100000&code)>0)printf("   DDU S-Link Not Ready\n");
      if((0x00300000&code)==0x00200000)printf("\n     DDU S-Link Stopped (backpressure)\n");
      printf("\n");
    }
    if((code&0x000F0000)>0){
      if((0x00080000&code)>0)printf("   DDU TMB Error\n");
      if((0x00040000&code)>0)printf("   DDU ALCT Error\n");
      if((0x00020000&code)>0)printf("   DDU Trigger Readout Wordcount Error\n");
      if((0x00010000&code)>0)printf("   DDU Trigger L1A Match Error\n");
      printf("\n");
    }
  }
}

void CSCDDUEventData::unpack_data(unsigned short *buf) {
  // just to calculate length
  unsigned short * inputBuf = buf;
  theData.clear();
  if (debug) std::cout << "CSCDDUEventData::unpack_data() is called" << std::endl;

  // decode DCC header (128 bits)
  if (dccData) {
    if (debug) std::cout << "unpacking dcc header..." << std::endl;
    memcpy(&theDCCHeader, buf, theDCCHeader.sizeInWords()*2);
    buf += theDCCHeader.sizeInWords();
  }

  if (debug) for (int i=0;i<4;i++) {
    printf("%d   %04x %04x %04x %04x\n",i,buf[4*i+3],buf[4*i+2],buf[4*i+1],buf[4*i]);
  }

  memcpy(&theDDUHeader, buf, theDDUHeader.sizeInWords()*2);

  if (debug) {
    std::cout << "size of ddu header in words = " << theDDUHeader.sizeInWords() << std::endl;
    std::cout << "sizeof(DDUHeader) = " << sizeof(theDDUHeader) << std::endl;
  }
  buf += theDDUHeader.sizeInWords();
  
  
  int i=-1;
 
  // we really don't want to copy CSCEventData's while filling the vec
  theData.clear();
  theData.reserve(theDDUHeader.ncsc());

  while( (((buf[0]&0xf000) == 0x9000)||((buf[0]&0xf000) == 0xa000)) 
          && (buf[3] != 0x8000)){
    ++i;
    if (debug) std::cout << std::endl << "unpack csc data loop started" << std::endl;
    theData.push_back(CSCEventData(buf));
    buf += (theData.back()).size();
    if (debug) {
      std::cout << "size of vector of cscDatas = " << theData.size() << std::endl;
      printf("%04x %04x %04x %04x\n",buf[3],buf[2],buf[1],buf[0]);
    }
  }


  if (debug) {
      std::cout << "unpacking ddu trailer " << std::endl;
      printf("%04x %04x %04x %04x\n",buf[3],buf[2],buf[1],buf[0]);
  }

  // decode ddu tail
  memcpy(&theDDUTrailer, buf, theDDUTrailer.sizeInWords()*2);
  if (debug) theDDUTrailer.check();
  errorstat=theDDUTrailer.errorstat();
  if (errorstat != 0) {
    std::cout << "+++ CSCDDUEventData warning: DDU Trailer errors = " << std::hex
         << errorstat << std::dec << " +++ " << std::endl;
    if (debug) decodeStatus(errorstat);
  }
   
  if (debug) std::cout << " Final errorstat " << std::hex << errorstat << std::endl;
  // the trailer counts in 64-bit words
  buf += theDDUTrailer.sizeInWords();

  //decode dcc trailer (128 bits)
  if (dccData) {
    if (debug) { std::cout<<"decoding DCC trailer" << std::endl;}
    memcpy(&theDCCTrailer, buf, theDCCTrailer.sizeInWords()*2);
    if (debug) std::cout << "checking DCC trailer " << theDCCTrailer.check() << std::endl; 
    buf += theDCCTrailer.sizeInWords();
    if (debug) std::cout << "checking DCC trailer " << theDCCTrailer.check() << std::endl;
  }

  theSizeInWords = buf - inputBuf;
}


bool CSCDDUEventData::check() const {
  // the trailer counts in 64-bit words
  if (debug) {
    std::cout << "size in Words () = " << std::dec << sizeInWords() << std::endl;
    std::cout << "wordcount = " << std::dec << theDDUTrailer.wordcount()*4 << 
      std::hex<< std::endl;
    //cout << "DDUHeader check " << hex <<theDDUHeader.check() << endl;
    //cout << "DDUTrailer.check " << theDDUTrailer.check() << endl;
  }

  return //(sizeInWords() == theDDUTrailer.word_count * 4) &&
          theDDUHeader.check() && theDDUTrailer.check();
}


/*BitVector CSCDDUEventData::pack() {
  BitVector result;

  BitVector dduHeader((const unsigned *) &theDDUHeader, 
                      theDDUHeader.sizeInWords()*16);
  result.assign(result.nBits(), dduHeader.nBits(), dduHeader);

  for(unsigned int i = 0; i < theData.size(); ++i) {
    BitVector eventData = theData[i].packVector();
    result.assign(result.nBits(), eventData.nBits(), eventData);
  }

  BitVector dduTrailer((const unsigned *) &theDDUTrailer,
                      theDDUTrailer.sizeInWords()*16);
  result.assign(result.nBits(), dduTrailer.nBits(), dduTrailer);

  return result;
}
*/
