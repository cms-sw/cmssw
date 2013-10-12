#ifndef L1TYELLOWFIRMWAREIMP_H
#define L1TYELLOWFIRMWAREIMP_H

#include "L1Trigger/L1TYellow/interface/YellowFirmware.h"
#include "L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h"

//
//  YellowFirmwareImp:
//
//    Class declarations for entire collection of version dependent firmware.
//
//    All Firmware versions satisfy the YellowFirmware interface.
//

namespace l1t {

  
  class YellowFirmwareImp1 : public YellowFirmware {
  public:
    YellowFirmwareImp1(const YellowParams & dbPars);
    virtual ~YellowFirmwareImp1();
    virtual void processEvent(const YellowDigiCollection & input, YellowOutputCollection & out);
  private:
    YellowParams const & db;
  };

  class YellowFirmwareImp2 : public YellowFirmware {
  public:
    YellowFirmwareImp2(const YellowParams & dbPars);
    virtual ~YellowFirmwareImp2();
    virtual void processEvent(const YellowDigiCollection & input, YellowOutputCollection & out);
  private:
    YellowParams const & db;
  };
  
} // namespace

#endif


