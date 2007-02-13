
#include "EventFilter/GctRawToDigi/src/GctBlockConverter.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
//#include "EventFilter/GctRawToDigi/interface/L1GctInternalObject.h"


#include <iostream>

using std::cout;
using std::endl;

GctBlockConverter::GctBlockConverter() { 
  blockLength_[104] = 4;
  blockLength_[105] = 16;
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

template< >
void GctBlockConverter::convertBlock(const unsigned char * data, unsigned id, std::vector<L1GctEmCand>* coll) {

  for (int i=0; i<blockLength(id); i++) {
    L1GctEmCand em0((data[i*4] + data[i*4+1]<<8), true );
    L1GctEmCand em1((data[i*4+2] + data[i*4+3]<<8), true );
    coll->push_back( em0 );
    coll->push_back( em1 );
  }

}

