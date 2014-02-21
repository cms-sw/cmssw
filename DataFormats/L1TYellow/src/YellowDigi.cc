///
/// \class l1t::YellowDigi
///
/// Description: See header file.
///
/// Implementation:
///
/// \author: Michael Mulhearn - UC Davis
///

#include "DataFormats/L1TYellow/interface/YellowDigi.h"

using namespace std;
using namespace l1t;

YellowDigi::YellowDigi() : m_et(0), m_yvar(0) { }

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
