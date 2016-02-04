///
/// \class l1t::MicroGMTRelativeIsolationCheckLUTFactory
///
/// \author: Thomas Reis
///
//
// This class implements the RelativeIsolationCheckLUT factory. Based on the firmware 
// version it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

using namespace edm;

namespace l1t {
  MicroGMTRelativeIsolationCheckLUTFactory::ReturnType
  MicroGMTRelativeIsolationCheckLUTFactory::create(const std::string& filename, const int fwVersion) {
    ReturnType p;

    switch (fwVersion) {
      case 1:
        p = ReturnType(new MicroGMTRelativeIsolationCheckLUT(filename));
        break;
      default:
        LogError("MicroGMTRelativeIsolationCheckLUTFactory") << "Invalid firmware version requested: " << fwVersion;
    }
    return p;
  }
}
