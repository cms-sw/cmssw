///
/// \class l1t::Stage1Layer2FirmwareFactory
///
///
/// \author: R. Alex Barbieri
///

//
// This class implments the firmware factory. Based on the firmware version
// in the configuration, it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2MainProcessorFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2FirmwareFactory.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

using namespace std;
using namespace edm;

namespace l1t {

  Stage1Layer2FirmwareFactory::ReturnType
  Stage1Layer2FirmwareFactory::create(const int m_fwv ,CaloParamsHelper* dbPars){
    ReturnType p;
    //unsigned fwv = m_fwv.firmwareVersion();
    //unsigned fwv = 1;

    // It is up to developers to choose when a new concrete firmware
    // implementation is needed. In this example, Imp1 handles FW
    // versions 1 and 2, while Imp2 handles FW version 3.

    switch (m_fwv){
    case 1:
    case 2:
    case 3:
      p = ReturnType(new Stage1Layer2MainProcessorFirmwareImp1(m_fwv, dbPars));
      break;
    default:
      // Invalid Firmware, log an error:
      LogError("l1t|stage 1 jets") << "Invalid firmware version requested: " << m_fwv << "\n";
      break;
    }

    return p;
  }

}
