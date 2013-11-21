///
/// Description: Firmware headers
///
/// Implementation:
///    Collects concrete firmware implmentations.
///
/// \author: Michael Mulhearn - UC Davis
///

//
//  This header file contains the class definitions for all of the concrete
//  implementations of the firmware interface.  The YellowFirmwareFactory
//  selects the appropriate implementation based on the firmware version in the
//  configuration.
//

#ifndef L1TYELLOWFIRMWAREIMP_H
#define L1TYELLOWFIRMWAREIMP_H

#include "L1Trigger/L1TYellow/interface/YellowFirmware.h"
#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class YellowFirmwareImp1 : public YellowFirmware {
  public:
    YellowFirmwareImp1(const YellowParams & dbPars);
    virtual ~YellowFirmwareImp1();
    virtual void processEvent(const YellowDigiCollection & input, YellowOutputCollection & out);
  private:
    YellowParams const & db;
  };

  // Imp2 is for v3
  class YellowFirmwareImp2 : public YellowFirmware {
  public:
    YellowFirmwareImp2(const YellowParams & dbPars);
    virtual ~YellowFirmwareImp2();
    virtual void processEvent(const YellowDigiCollection & input, YellowOutputCollection & out);
  private:
    YellowParams const & db;
  };
  
}

#endif


