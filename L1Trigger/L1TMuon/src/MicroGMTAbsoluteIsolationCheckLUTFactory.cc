///
/// \class l1t::MicroGMTAbsoluteIsolationCheckLUTFactory
///
/// \author: Thomas Reis
///
//
// This class implements the AbsoluteIsolationCheckLUT factory. Based on the firmware 
// version it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

using namespace edm;

namespace l1t {
  MicroGMTAbsoluteIsolationCheckLUTFactory::ReturnType
  MicroGMTAbsoluteIsolationCheckLUTFactory::create(const std::string& filename, const int fwVersion) {
    ReturnType p;

    if (fwVersion >= 1) {
      p = ReturnType(new MicroGMTAbsoluteIsolationCheckLUT(filename));
    } else {
      LogError("MicroGMTAbsoluteIsolationCheckLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }

  MicroGMTAbsoluteIsolationCheckLUTFactory::ReturnType
  MicroGMTAbsoluteIsolationCheckLUTFactory::create(l1t::LUT* lut, const int fwVersion) {
    ReturnType p;

    if (fwVersion >= 1) {
        p = ReturnType(new MicroGMTAbsoluteIsolationCheckLUT(lut));
    } else {
        LogError("MicroGMTAbsoluteIsolationCheckLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }
}
