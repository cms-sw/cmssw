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
  MicroGMTExtrapolationLUTFactory::create(const std::string& filename, const int fwVersion) {
    ReturnType p;

    switch (fwVersion) {
      case 1:
        p = ReturnType(new MicroGMTExtrapolationLUT(filename));
        break;
      default:
        LogError("MicroGMTExtrapolationLUTFactory") << "Invalid firmware version requested: " << fwVersion;
    }
    return p;
  }
}
