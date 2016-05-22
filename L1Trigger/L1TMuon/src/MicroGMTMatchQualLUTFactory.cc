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
  MicroGMTMatchQualLUTFactory::ReturnType MicroGMTMatchQualLUTFactory::create(
      const std::string& filename, const double maxDR, const double fEta, const double fEtaCoarse,
      const double fPhi, cancel_t cancelType, const int fwVersion) {
    ReturnType p;
  
    if (fwVersion == 1) {
      p = ReturnType(new MicroGMTMatchQualSimpleLUT(
          filename, maxDR, fEtaCoarse, fPhi, cancelType));
    } else if (fwVersion >= 0x2020000) {
      p = ReturnType(new MicroGMTMatchQualFineLUT(
          filename, maxDR, fEta, fEtaCoarse, fPhi, cancelType));
    } else {
      LogError("MicroGMTMatchQualLUTFactory")
          << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }

  MicroGMTMatchQualLUTFactory::ReturnType
  MicroGMTMatchQualLUTFactory::create(l1t::LUT* lut, cancel_t cancelType, const int fwVersion) {
    ReturnType p;

    if (fwVersion == 1) {
      p = ReturnType(new MicroGMTMatchQualSimpleLUT(lut, cancelType));
    } else if (fwVersion >= 0x2020000) {
      p = ReturnType(new MicroGMTMatchQualFineLUT(lut, cancelType));
    } else {
        LogError("MicroGMTMatchQualLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }
}
