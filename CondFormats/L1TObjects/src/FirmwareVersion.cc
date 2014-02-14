///
/// \class l1t::FirmwareVersion
///
/// Description: see header file
///
/// Implementation:
///    
/// \author: Michael Mulhearn - UC Davis
///

#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

using namespace l1t;

void FirmwareVersion::print(std::ostream& out) const {
  out << "firmware version:  " << firmwareVersion() << "\n";
}
