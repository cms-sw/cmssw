
#include "EventFilter/GctRawToDigi/src/GctDaqRecord.h"

using std::vector;
using std::ostream;
using std::cout;
using std::endl;


GctDaqRecord::GctDaqRecord() :
  header_(8),
  footer_(8)
{ }


GctDaqRecord::GctDaqRecord(const unsigned char * data, const unsigned int size) :
  header_(8),
  footer_(8)
 {

   cout << "Constructing a GctDaqRecord. Size=" << size << endl;


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
     //     GctBlock block(&data[i]);
     //      if (block.length()==0) { i++; }
     //      else { i=i+block.length(); }
     //     blocks_.push_back(block);
   }

}


GctDaqRecord::~GctDaqRecord() {

  //  blocks_.clear();

}

ostream& operator<<(ostream& os, const GctDaqRecord& e) {
  
  os << "Event ID   " << std::hex << e.id() << endl;
  os << "Event type " << std::hex << e.l1Type() << endl;
  os << "BX ID      " << std::hex << e.bcId() << endl;
  os << "Source ID  " << std::hex << e.sourceId() << endl;
  
  vector<GctBlockHeader>::const_iterator b;
  for (b=e.blockHeaders_.begin(); b!=e.blockHeaders_.end(); b++) {
    os << *b;
  }

  return os;

}
