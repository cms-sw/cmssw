
#include "EventFilter/GctRawToDigi/src/GctBlockConverter.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <iostream>

#define CALL_GCT_CONVERT_FN(object,ptrToMember)  ((object).*(ptrToMember))

using std::cout;
using std::endl;

GctBlockConverter::GctBlockConverter() {

  // setup block length map
  blockLength_[0x68] = 4;   // ConcElec: Output to Global Trigger
  blockLength_[0x69] = 16;  // ConcElec: Sort Input
  blockLength_[0x80] = 20;  // Leaf-U1, Elec, NegEta, Sort Input
  blockLength_[0x81] = 15;  // Leaf-U1, Elec, NegEta, Raw Input
  blockLength_[0x83] = 4;   // Leaf-U1, Elec, NegEta, Sort Output
  blockLength_[0x88] = 16;  // Leaf-U2, Elec, NegEta, Sort Input
  blockLength_[0x89] = 12;  // Leaf-U2, Elec, NegEta, Raw Input
  blockLength_[0x8B] = 4;   // Leaf-U2, Elec, NegEta, Sort Output

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
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x80) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x81) :
    blockToRctEmCand(data, id, nSamples);
    break;
  case (0x83) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x88) :
    blockToGctInternEmCand(data, id, nSamples);
    break;
  case (0x89) :
    blockToRctEmCand(data, id, nSamples);
    break;
  case (0x8B) :
    blockToGctInternEmCand(data, id, nSamples);
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
      gctIsoEm_->push_back( L1GctEmCand(data[offset]   + (data[offset+1]<<8), true) );
      gctIsoEm_->push_back( L1GctEmCand(data[offset+2] + (data[offset+3]<<8), true) );
    }
    else {
      gctNonIsoEm_->push_back( L1GctEmCand(data[offset] + (data[offset+1]<<8), false) );
      gctNonIsoEm_->push_back( L1GctEmCand(data[offset+2] + (data[offset+3]<<8), false) );
    }
  }  
}

// ConcElec: Sort Input
// re-arrange intermediate data to match GT output format
void GctBlockConverter::blockToGctInternEmCand(const unsigned char * data, unsigned id, unsigned nSamples) {
  for (int i=0; i<blockLength(id)*nSamples; i=i+nSamples) {  // temporarily just take 0th time sample
    unsigned offset = i*4*nSamples;
    uint16_t w0 = data[offset]   + (data[offset+1]<<8); 
    uint16_t w1 = data[offset+2] + (data[offset+3]<<8);
    gctInternEm_->push_back( L1GctInternEmCand(w0, i > 7, id, 2*i/nSamples) );
    gctInternEm_->push_back( L1GctInternEmCand(w1, i > 7, id, 2*(i/nSamples)+1) );
  }
}

// Leaf-U2, Elec, NegEta, Raw Input
// this the last f*cking time I deal the RCT bit assignment travesty!!!
void GctBlockConverter::blockToRctEmCand(const unsigned char * data, unsigned id, unsigned nSamples) {
  
  uint16_t d[6]; // index = source card output * 2 + cycle
  
  unsigned first=0;
  unsigned last=0;
  if (id==0x81) { 
    first = 4;
    last = 8;
  }
  else if (id==0x89) { 
    first = 0;
    last = 3;
  }

  // loop over crates
  for (int crate=first; crate<=last; crate++) {
    
    unsigned offset = (crate-first)*12*nSamples; // just get 0th time sample for now
    
    // read 16 bit words
    for (int j=0; j<6; j++) {
      d[j] = data[offset+(2*nSamples*j)] + (data[offset+(2*nSamples*j)+1]<<8);
    }

    // create candidates and add to collections
    rctEm_->push_back( L1CaloEmCand( d[0] & 0x3ff, crate, true) );
    unsigned em = ((d[0] & 0x3800)>>10) + ((d[2] & 0x7800)>>7) + ((d[4] & 0x3800)>>3);
    rctEm_->push_back( L1CaloEmCand(   em & 0x3ff, crate, true) );
    rctEm_->push_back( L1CaloEmCand( d[1] & 0x3ff, crate, true) );
    em = ((d[1] & 0x3800)>>10) + ((d[3] & 0x7800)>>7) + ((d[5] & 0x3800)>>3);
    rctEm_->push_back( L1CaloEmCand(   em & 0x3ff, crate, true) );
    rctEm_->push_back( L1CaloEmCand( d[2] & 0x3ff, crate, false) );
    rctEm_->push_back( L1CaloEmCand( d[4] & 0x3ff, crate, false) );
    rctEm_->push_back( L1CaloEmCand( d[3] & 0x3ff, crate, false) );
    rctEm_->push_back( L1CaloEmCand( d[5] & 0x3ff, crate, false) );
  }

}
