///
/// \class l1t::MicroGMTExtrapolationLUTFactory
///
/// \author: Thomas Reis
///
//
// This class implements the ExtrapolationLUT factory. Based on the firmware 
// version it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

using namespace edm;

namespace l1t {
  MicroGMTExtrapolationLUTFactory::ReturnType
  MicroGMTExtrapolationLUTFactory::create(const std::string& filename, const int type, const int fwVersion) {
    ReturnType p;

    if (fwVersion >= 1) {
      p = ReturnType(new MicroGMTExtrapolationLUT(filename, type));
    } else {
      LogError("MicroGMTExtrapolationLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }

  MicroGMTExtrapolationLUTFactory::ReturnType
  MicroGMTExtrapolationLUTFactory::create(l1t::LUT* lut, const int type, const int fwVersion) {
    ReturnType p;

    if (fwVersion >= 1) {
      p = ReturnType(new MicroGMTExtrapolationLUT(lut, type));
    } else {
      LogError("MicroGMTExtrapolationLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }
}
