
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
  //  convertFn_[0x68] = &GctBlockConverter::wordToGctEmCand;
  //  convertFn_[0x69] = &GctBlockConverter::wordToGctInterEmCand;

}

GctBlockConverter::~GctBlockConverter() { }

// recognise block ID
bool GctBlockConverter::validBlock(unsigned id) {
  return ( blockLength_.find(id) != blockLength_.end() );
}

// return block length in 32-bit words
unsigned GctBlockConverter::blockLength(unsigned id) {
  return blockLength_.find(id)->second;
}

// conversion
void GctBlockConverter::convertBlock(const unsigned char * data, unsigned id, unsigned nSamples) {

  switch (id) {
  case (0x68) : 
      blockToGctEmCand(data, id, nSamples);
      break;
  case (0x69) : 
      blockToGctInterEmCand(data, id, nSamples);
      break;
  case (0x89) : 
      blockToRctEmCand(data, id, nSamples);
      break;
  default:
    std::cout << "Trying to unpack an identified block!" << std::endl;
    break;
  }

}



// ConcElec: Output to Global Trigger
// formats defined for GT output, no need to change
// NB - need to get GCT output digis in the right place in the record!
void GctBlockConverter::blockToGctEmCand(const unsigned char * data, unsigned id, unsigned nSamples) {
  for (int i=0; i<blockLength(id)*nSamples; i=i+nSamples) {  // temporarily just take 0th time sample
    unsigned offset = i*4*nSamples;
    bool iso = (i > 1);
    if (i > 1) {
      gctIsoEm->push_back( L1GctEmCand(data[offset]   + (data[offset+1]<<8), true) );
      gctIsoEm->push_back( L1GctEmCand(data[offset+2] + (data[offset+3]<<8), true) );
    }
    else {
      gctNonIsoEm->push_back( L1GctEmCand(data[offset] + (data[offset+1]<<8), false) );
      gctNonIsoEm->push_back( L1GctEmCand(data[offset+2] + (data[offset+3]<<8), false) );
    }
  }  
}

// ConcElec: Sort Input
// re-arrange intermediate data to match GT output format
void GctBlockConverter::blockToGctInterEmCand(const unsigned char * data, unsigned id, unsigned nSamples) {
  for (int i=0; i<blockLength(id)*nSamples; i=i+nSamples) {  // temporarily just take 0th time sample
    unsigned offset = i*4*nSamples;
    uint16_t w0 = data[offset]   + (data[offset+1]<<8); 
    uint16_t w1 = data[offset+2] + (data[offset+3]<<8);
    w0 = (w0 & 0x1ff) + ((w0 & 0xfc00)>>1);
    w1 = (w1 & 0x1ff) + ((w1 & 0xfc00)>>1);
    gctInterEm->push_back( L1GctEmCand(w0, i > 7) );
    gctInterEm->push_back( L1GctEmCand(w1, i > 7) );
  }
}

// Leaf-U2, Elec, NegEta, Raw Input
// this the last f*cking time I deal the RCT bit assignment travesty!!!
void GctBlockConverter::blockToRctEmCand(const unsigned char * data, unsigned id, unsigned nSamples) {
  
  uint16_t d[6]; // index = source card output * 2 + cycle
  
  for (int iRct=0; iRct<blockLength(id)*nSamples; iRct=iRct+(nSamples*3)) {  // temporarily just take 0th time sample
    unsigned offset = iRct*4*3*nSamples;  // 4 bytes per 32-bits, 3 SC outputs per RCT crate
    for (int j=0; j<7; j++) {
      d[j] = data[offset+(j/2)] + (data[offset+(j/2)+1]<<8); 
    }
    
    rctEm->push_back( L1CaloEmCand( d[0] & 0x1ff, iRct, true) );
    rctEm->push_back( L1CaloEmCand( ((d[0] & 0x3800)>>10) + ((d[2] & 0x7800)>>7) + ((d[4] & 0x3800)>>3), iRct, true) );
    rctEm->push_back( L1CaloEmCand( d[1] & 0x1ff, iRct, true) );
    rctEm->push_back( L1CaloEmCand( ((d[1] & 0x3800)>>10) + ((d[3] & 0x7800)>>7) + ((d[5] & 0x3800)>>3), iRct, true) );
    rctEm->push_back( L1CaloEmCand( d[2] & 0x1ff, iRct, true) );
    rctEm->push_back( L1CaloEmCand( d[4] & 0x1ff, iRct, true) );
    rctEm->push_back( L1CaloEmCand( d[3] & 0x1ff, iRct, true) );
    rctEm->push_back( L1CaloEmCand( d[5] & 0x1ff, iRct, true) );

  }  

}
