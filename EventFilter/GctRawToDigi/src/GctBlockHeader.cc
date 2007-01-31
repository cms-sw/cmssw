
#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

using std::ostream;
using std::endl;
using std::hex;
using std::dec;

GctBlockHeader::GctBlockHeader(const unsigned char * data)
{ 
  for (int i=0; i<4; i++) {
    d.push_back(data[i]);
  }
}

GctBlockHeader::~GctBlockHeader() {

}

unsigned int GctBlockHeader::blockLength() const {
  return 0;
}

ostream& operator<<(ostream& os, const GctBlockHeader& h) {
  os << "ID " << hex << h.id() << " : Samples " << h.nSamples() << " : BX " << h.bcId() << " : Event " << h.eventId() << " : Length " << h.blockLength() << dec << endl;
  return os;
}
