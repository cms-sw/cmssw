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
  MicroGMTRankPtQualLUTFactory::create(const std::string& filename, const int fwVersion, const unsigned ptFactor, const unsigned qualFactor) {
    ReturnType p;

    if (fwVersion >= 1) {
      p = ReturnType(new MicroGMTRankPtQualLUT(filename, ptFactor, qualFactor));
    } else {
      LogError("MicroGMTRankPtQualLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }

  MicroGMTRankPtQualLUTFactory::ReturnType
  MicroGMTRankPtQualLUTFactory::create(l1t::LUT* lut, const int fwVersion) {
    ReturnType p;

    if (fwVersion >= 1) {
      p = ReturnType(new MicroGMTRankPtQualLUT(lut));
    } else {
      LogError("MicroGMTRankPtQualLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }
}
