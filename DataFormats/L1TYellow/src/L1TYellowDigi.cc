#include "DataFormats/L1TYellow/interface/L1TYellowDigi.h"

using namespace std;
//using namespace l1t;

// default constructor
L1TYellowDigi::L1TYellowDigi() : m_data(0) { }

// destructor:
L1TYellowDigi::~L1TYellowDigi(){ }

//namespace l1t {
  // print to stream
  std::ostream& operator << (std::ostream& os, const L1TYellowDigi& x) {
    os << "L1TYellowDigi:";
    os << " Et=" << x.et();
    os << " raw=" << x.m_data << " ";
    return os;
  }
//}
