#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"
#include "YellowFirmwareImp.h"

using namespace std;

namespace l1t {

  YellowFirmwareFactory::ReturnType
  YellowFirmwareFactory::create(const YellowParams & dbPars){
    ReturnType p;
    unsigned fwv = dbPars.firmwareVersion();
    cout << "Using firmware version:  " << fwv << "\n";

    // It is up to developers to choose when a new concrete firmware
    // implementation is needed.  In this example, Imp1 handles FW
    // versions 1 and 2, while Imp2 handles FW version 3.

    switch (fwv){
    case 1:
      p = ReturnType(new YellowFirmwareImp1(dbPars));  
      break;
    case 2:
      p = ReturnType(new YellowFirmwareImp1(dbPars));  
      break;
    case 3:
      p = ReturnType(new YellowFirmwareImp2(dbPars));  
      break;
    default:
      // Invalid Firmware, log an error:
      break;
    }

    return p;
  }

}
