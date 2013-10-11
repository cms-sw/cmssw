#include "DataFormats/L1TYellow/interface/YellowDigi.h"

// within .cc files, it is OK to introduce an entire namespace:
using namespace std;
using namespace l1t;

// default constructor
YellowDigi::YellowDigi() : m_et(0), m_yvar(0) { }

// destructor:
YellowDigi::~YellowDigi(){}

namespace l1t {
  // print to stream
  std::ostream& operator << (std::ostream& os, const YellowDigi& x) {
    os << "l1t::YellowDigi:";
    os << " Et=" << x.et();
    os << " Y-var==" << x.yvar() << " ";
    return os;
  }
}
