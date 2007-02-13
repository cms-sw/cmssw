

#include "EventFilter/GctRawToDigi/src/GctBlock.h"

using std::ostream;
using std::endl;
using std::hex;
using std::dec;


GctBlock::GctBlock(const unsigned char * data, unsigned length) {
  for (unsigned i=0; i<length; i++) {
    unsigned word = data[4+i] + data[5+i]<<8 + data[6+i]<<16 + data[7+i]<<24;
    data_.push_back(word);
  }
}

GctBlock::~GctBlock() {

}

ostream& operator<<(ostream& os, const GctBlock& b) {
  //  os << "Block :" << b.head_;

  os << hex;
  for (unsigned i=0; i < b.data_.size(); i++) {
    os << b.data_[i] << endl;
  }
  os << endl;

  return os;
}
