
#include "EventFilter/GctRawToDigi/src/GctEvent.h"

using std::vector;
using std::ostream;
using std::cout;
using std::endl;


GctEvent::GctEvent() :
  header_(8),
  footer_(8)
{ }


GctEvent::GctEvent(const unsigned char * data, const unsigned int size) :
  header_(8),
  footer_(8)
 {

   cout << "Constructing a GctEvent. Size=" << size << endl;


  // store event header
   for (unsigned int i=0; i<8; i++) {
     header_[i] = data[i];
   }

   for (unsigned i=0; i<8; i++) {
     footer_[9-i] = data[size+1-i];
   }


   // store blocks
   unsigned i=8;
   while(i<size-8) {
     GctBlock block(&data[i]);
     if (block.length()==0) { i++; }
     else { i=i+block.length(); }
     blocks_.push_back(block);
   }

}


GctEvent::~GctEvent() {

  //  blocks_.clear();

}

ostream& operator<<(ostream& os, const GctEvent& e) {
  
  os << "Event ID   " << std::hex << e.id() << endl;
  os << "Event type " << std::hex << e.l1Type() << endl;
  os << "BX ID      " << std::hex << e.bcId() << endl;
  os << "Source ID  " << std::hex << e.sourceId() << endl;
  
  vector<GctBlock>::const_iterator b;
  for (b=e.blocks_.begin(); b!=e.blocks_.end(); b++) {
    os << *b;
  }

  return os;

}
