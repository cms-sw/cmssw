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

    switch (fwVersion) {
      case 1:
        p = ReturnType(new MicroGMTAbsoluteIsolationCheckLUT(filename));
        break;
      default:
        LogError("MicroGMTAbsoluteIsolationCheckLUTFactory") << "Invalid firmware version requested: " << fwVersion;
    }
    return p;
  }
}
