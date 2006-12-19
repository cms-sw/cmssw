
#include "EventFilter/GctRawToDigi/src/GctEvent.h"

using std::vector;
using std::ostream;
using std::endl;


GctEvent::GctEvent(const unsigned char * data, const unsigned int size) {

  // store event header
  for (unsigned i=0; i<8; i++) {
    header_[i] = data[i];
  }

  for (unsigned i=1; i<=8; i++) {
    footer_[8-i] = data[size-i];
  }

  // store blocks
  for (unsigned i=8; i<size-8; ) {

    GctBlock block(&data[i]);
    i = i+block.length();

    blocks_.push_back(block);

  }

}


GctEvent::~GctEvent() {

  blocks_.clear();

}

ostream& operator<<(ostream& os, const GctEvent& e) {
  
  os << "Event ID  " << e.id() << endl;
  os << "L1 type   " << e.l1Type() << endl;
  os << "BX ID     " << e.bcId() << endl;
  os << "Source ID " << e.sourceId() << endl;
  

}
