#include "DataFormats/L1TYellow/interface/L1TYellowOutput.h"

using namespace std;
//using namespace l1t;

// default constructor
L1TYellowOutput::L1TYellowOutput() : m_data(0) { }

// destructor:
L1TYellowOutput::~L1TYellowOutput(){ }

//namespace l1t {
  // print to stream
  std::ostream& operator << (std::ostream& os, const L1TYellowOutput& x) {
    os << "L1TYellowOutput:";
    os << " Et=" << x.et();
    os << " raw=" << x.m_data << " ";
    return os;
  }
//}
