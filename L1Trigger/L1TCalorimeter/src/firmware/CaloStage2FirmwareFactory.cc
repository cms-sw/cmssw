///
/// \class l1t::CaloStage1FirmwareFactory
///
///
/// \author: Jim Brooke
///

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2MainProcessorFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2FirmwareFactory.h"


using namespace std;
using namespace edm;

l1t::CaloStage2FirmwareFactory::ReturnType
l1t::CaloStage2FirmwareFactory::create(const FirmwareVersion & fwv) {
  ReturnType p;
  unsigned v = fwv.firmwareVersion();
  
  switch (v){
  case 1:
    p = ReturnType(new CaloStage2MainProcessorFirmwareImp1(fwv));
      break;
  default:
    // Invalid Firmware, log an error:
    LogError("l1t|stage 2 calo") << "Invalid firmware version requested: " << v << "\n";
    break;
  }
  
  return p;

}
