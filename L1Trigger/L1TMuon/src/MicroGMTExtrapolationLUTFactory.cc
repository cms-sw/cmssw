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

    int outWidth = 3;
    if (type == MicroGMTConfiguration::ETA_OUT) {
      outWidth = 4;
    }

    if (fwVersion >= 1 && fwVersion < 0x4010000) {
      p = std::make_shared<l1t::MicroGMTExtrapolationLUT>(filename, outWidth, 6, 6);
    } else if (fwVersion >= 0x4010000) {
      p = std::make_shared<l1t::MicroGMTExtrapolationLUT>(filename, 4, 5, 7);
    } else {
      LogError("MicroGMTExtrapolationLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }

  MicroGMTExtrapolationLUTFactory::ReturnType
  MicroGMTExtrapolationLUTFactory::create(l1t::LUT* lut, const int type, const int fwVersion) {
    ReturnType p;

    int outWidth = 3;
    if (type == MicroGMTConfiguration::ETA_OUT) {
      outWidth = 4;
    }

    if (fwVersion >= 1 && fwVersion < 0x4010000) {
      p = std::make_shared<l1t::MicroGMTExtrapolationLUT>(lut, outWidth, 6, 6);
    } else if (fwVersion >= 0x4010000) {
      p = std::make_shared<l1t::MicroGMTExtrapolationLUT>(lut, 4, 5, 7);
    } else {
      LogError("MicroGMTExtrapolationLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }
}
