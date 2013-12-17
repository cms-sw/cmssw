///
/// \class l1t::YellowOutput
///
/// Description: See header file.
///
/// Implementation:
///
/// \author: Michael Mulhearn - UC Davis
///

#include "DataFormats/L1TYellow/interface/YellowOutput.h"

// within .cc files, it is OK to introduce an entire namespace:
using namespace std;
using namespace l1t;

// default constructor
YellowOutput::YellowOutput() : m_et(0), m_yvar(0) { }

// destructor:
YellowOutput::~YellowOutput(){}

namespace l1t {
  // print to stream
  std::ostream& operator << (std::ostream& os, const YellowOutput& x) {
    os << "l1t::YellowOutput:";
    os << " Et=" << x.et();
    os << " Y-var==" << x.yvar() << " ";
    return os;
  }
}
