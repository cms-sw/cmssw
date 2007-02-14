
#include "EventFilter/GctRawToDigi/src/GctBlockConverter.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
//#include "EventFilter/GctRawToDigi/interface/L1GctInternalObject.h"


#include <iostream>

#define CALL_GCT_CONVERT_FN(object,ptrToMember)  ((object).*(ptrToMember))

using std::cout;
using std::endl;

GctBlockConverter::GctBlockConverter() {

  // setup block length map
  blockLength_[0x68] = 4;
  blockLength_[0x69] = 16;
  blockLength_[0x88] = 16;
  blockLength_[0x89] = 12;
  blockLength_[0x8B] = 4;

  // setup converter fn map
  convertFn_[0x68] = &GctBlockConverter::wordToGctEmCand;
  convertFn_[0x69] = &GctBlockConverter::wordToGctInterEmCand;

}

GctBlockConverter::~GctBlockConverter() { }

// recognise block ID
bool GctBlockConverter::validBlock(unsigned id) {
  return ( blockLength_.find(id)->second != 0 );
}

// return block length in 32-bit words
unsigned GctBlockConverter::blockLength(unsigned id) {
  return blockLength_.find(id)->second;
}

// conversion
void GctBlockConverter::convertBlock(const unsigned char * data, unsigned id) {

  for (int i=0; i<blockLength(id); i++) {

    uint16_t w0 = data[i*4]   + data[i*4+1]<<8;  
    uint16_t w1 = data[i*4+2] + data[i*4+3]<<8;

    if (id==0x68) {  
      wordToGctEmCand(w0, w1, i);
    }
    else if (id==0x69) {
      wordToGctInterEmCand(w0, w1, i);      
    }

  }

}

// ConcElec: Output to Global Trigger
void GctBlockConverter::wordToGctEmCand(uint16_t w0, uint16_t w1, int i) {
  // formats defined for GT output, no need to change
  bool iso = (i > 1);
  if (iso) {
    gctIsoEm->push_back( L1GctEmCand(w0, iso) );
    gctIsoEm->push_back( L1GctEmCand(w1, iso) );
  }
  else {
    gctNonIsoEm->push_back( L1GctEmCand(w0, iso) );
    gctNonIsoEm->push_back( L1GctEmCand(w1, iso) );
  }
}

// ConcElec: Sort Input
void GctBlockConverter::wordToGctInterEmCand(uint16_t w0, uint16_t w1, int i) {
  // re-arrange intermediate data to match GT output format
  w0 = (w0 & 0x1ff) + (w0 & 0xfc00)>>1;
  w1 = (w1 & 0x1ff) + (w1 & 0xfc00)>>1;
  gctInterEm->push_back( L1GctEmCand(w0, i > 7) );
  gctInterEm->push_back( L1GctEmCand(w1, i > 7) );
}
