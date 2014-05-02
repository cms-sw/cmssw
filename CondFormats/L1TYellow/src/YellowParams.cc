///
/// \class l1t::YellowParams
///
/// Description: see header file
///
/// Implementation:
///    
/// \author: Michael Mulhearn - UC Davis
///

#include "CondFormats/L1TYellow/interface/YellowParams.h"

using namespace l1t;

void YellowParams::print(std::ostream& out) const {
  out << "firmware version:  " << firmwareVersion() << "\n";
  out << "paramA:  " << paramA() << "\n";
  out << "paramB:  " << paramB() << "\n";
  out << "paramC:  " << paramC() << "\n";
}
