

#include "EventFilter/GctRawToDigi/src/GctBlock.h"

using std::ostream;
using std::endl;
using std::hex;
using std::dec;


GctBlock::GctBlock(const unsigned char * data) :
  head(GctBlockHeader(data))
{ 
  for (unsigned i=0; i<head.blockLength(); i++) {
    d.push_back(data[i+4]); // +4 to get past header
  }  
}

GctBlock::~GctBlock() {

}

ostream& operator<<(ostream& os, const GctBlock& b) {
  os << "Block :" << b.head;
  for (unsigned i=0; i<(b.head.blockLength()); i=i+4) {
    int val = b.d[i] + (b.d[i+1]<<8) + (b.d[i+2]<<16) + (b.d[i+3]<<24); 
    os << hex << val << dec << endl;
  }
  return os;
}
