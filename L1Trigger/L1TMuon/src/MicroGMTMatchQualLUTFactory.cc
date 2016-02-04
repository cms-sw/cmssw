///
/// \class l1t::MicroGMTMatchQualLUTFactory
///
/// \author: Thomas Reis
///
//
// This class implements the MatchQualLUT factory. Based on the firmware 
// version it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

using namespace edm;

namespace l1t {
  MicroGMTMatchQualLUTFactory::ReturnType
  MicroGMTMatchQualLUTFactory::create(const std::string& filename, cancel_t cancelType, const int fwVersion) {
    ReturnType p;

    switch (fwVersion) {
      case 1:
        p = ReturnType(new MicroGMTMatchQualLUT(filename, cancelType));
        break;
      default:
        LogError("MicroGMTMatchQualLUTFactory") << "Invalid firmware version requested: " << fwVersion;
    }
    return p;
  }
}
