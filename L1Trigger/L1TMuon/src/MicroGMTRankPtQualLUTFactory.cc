///
/// \class l1t::MicroGMTRankPtQualLUTFactory
///
/// \author: Thomas Reis
///
//
// This class implements the RankPtQualLUT factory. Based on the firmware 
// version it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

using namespace edm;

namespace l1t {
  MicroGMTRankPtQualLUTFactory::ReturnType
  MicroGMTRankPtQualLUTFactory::create(const std::string& filename, const int fwVersion) {
    ReturnType p;

    switch (fwVersion) {
      case 1:
        p = ReturnType(new MicroGMTRankPtQualLUT(filename));
        break;
      default:
        LogError("MicroGMTRankPtQualLUTFactory") << "Invalid firmware version requested: " << fwVersion;
    }
    return p;
  }
}
