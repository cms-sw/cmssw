///
/// \class l1t::YellowFirmwareFactory
///
/// Description: Firmware factory class for the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to implement the firmare factory design pattern.
///
/// \author: Michael Mulhearn - UC Davis
///

//
//  See "L1Trigger/L1TYellow/interface/YellowFirmware.h" for a general
//  description of the firmware factory design pattern.
//
//  This class implments the firmware factory.  Based on the firmware version
//  in the configuration, it selects the appropriate concrete implementation.
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "YellowFirmwareImp.h"

#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"


using namespace std;
using namespace edm;

namespace l1t {

  YellowFirmwareFactory::ReturnType
  YellowFirmwareFactory::create(const YellowParams & dbPars){
    ReturnType p;
    unsigned fwv = dbPars.firmwareVersion();
    
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
      LogError("l1t|yellow") << "Invalid firmware version requested:  " << fwv << "\n";
      break;
    }

    return p;
  }

}
