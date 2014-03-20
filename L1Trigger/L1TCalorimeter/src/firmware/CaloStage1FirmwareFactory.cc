///
/// \class l1t::CaloStage1FirmwareFactory
///
///
/// \author: R. Alex Barbieri
///

//
// This class implments the firmware factory. Based on the firmware version
// in the configuration, it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1MainProcessorFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1FirmwareFactory.h"


using namespace std;
using namespace edm;

namespace l1t {

  CaloStage1FirmwareFactory::ReturnType
  CaloStage1FirmwareFactory::create(const FirmwareVersion & m_fwv /*,const CaloParams & dbPars*/){
    ReturnType p;
    unsigned fwv = m_fwv.firmwareVersion();
    //unsigned fwv = 1;

    // It is up to developers to choose when a new concrete firmware
    // implementation is needed. In this example, Imp1 handles FW
    // versions 1 and 2, while Imp2 handles FW version 3.

    switch (fwv){
    case 1:
      p = ReturnType(new CaloStage1MainProcessorFirmwareImp1(m_fwv /*dbPars*/));
      break;
    case 2:
      p = ReturnType(new CaloStage1MainProcessorFirmwareImp1(m_fwv /*dbPars*/));
      break;
    default:
      // Invalid Firmware, log an error:
      LogError("l1t|stage 1 jets") << "Invalid firmware version requested: " << fwv << "\n";
      break;
    }

    return p;
  }

}
