

#include "EventFilter/GctRawToDigi/src/GctBlock.h"

using std::ostream;
using std::endl;
using std::hex;
using std::dec;


GctBlock::GctBlock(const unsigned char * data) :
  head(GctBlockHeader(data)),
  d(&data[1])
{ 
  
}

GctBlock::~GctBlock() {

}

// get the data
std::vector<unsigned char> GctBlock::data() const { 
  std::vector<unsigned char> b;
  for (int i=0; i<head.length()*4; i++) {
    b.push_back(d[i]);
  }
  return b;
}


ostream& operator<<(ostream& os, const GctBlock& b) {
  os << "Block :" << b.head;
  for (int i=0; i<(b.head.length()*4); i=i+4) {
    int val = b.d[i] + (b.d[i+1]<<8) + (b.d[i+2]<<16) + (b.d[i+3]<<24); 
    os << hex << val << dec << endl;
  }
}
