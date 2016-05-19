///
/// \class l1t::MicroGMTCaloIndexSelectionLUTFactory
///
/// \author: Thomas Reis
///
//
// This class implements the CaloIndexSelectionLUT factory. Based on the firmware 
// version it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

using namespace edm;

namespace l1t {
  MicroGMTCaloIndexSelectionLUTFactory::ReturnType
  MicroGMTCaloIndexSelectionLUTFactory::create(const std::string& filename, const int type, const int fwVersion) {
    ReturnType p;

    if (fwVersion >= 1) {
      p = ReturnType(new MicroGMTCaloIndexSelectionLUT(filename, type));
    } else {
      LogError("MicroGMTCaloIndexSelectionLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }

  MicroGMTCaloIndexSelectionLUTFactory::ReturnType
  MicroGMTCaloIndexSelectionLUTFactory::create(l1t::LUT* lut, const int type, const int fwVersion) {
    ReturnType p;

    if (fwVersion >= 1) {
      p = ReturnType(new MicroGMTCaloIndexSelectionLUT(lut, type));
    } else {
      LogError("MicroGMTCaloIndexSelectionLUTFactory") << "Invalid firmware version requested: 0x" << std::hex << fwVersion << std::dec;
    }
    return p;
  }
}
