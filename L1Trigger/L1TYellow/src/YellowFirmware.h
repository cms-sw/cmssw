#ifndef L1TYellowFirmware_h
#define L1TYellowFirmware_h

#include "L1Trigger/L1TYellow/interface/YellowAlg.h"

//
//  YellowFirmware:
//
//    Class declarations for entire collection of version dependent firmware.
//
//    All Firmware versions satisfy the YellowAlg interface.
//

namespace l1t {
  
  class YellowAlg_v1 : public YellowAlg {
  public:
    YellowAlg_v1(const YellowParams & dbPars);
    virtual ~YellowAlg_v1();
    virtual void processEvent(const YellowDigiCollection & input, YellowOutputCollection & out);
  private:
    void do_something_specific_to_v1_firmware();
    YellowParams const & db;
    int data_specific_to_v1_firmware;
  };

  class YellowAlg_v2 : public YellowAlg {
  public:
    YellowAlg_v2(const YellowParams & dbPars);
    virtual ~YellowAlg_v2(){};
    virtual void processEvent(const YellowDigiCollection & input, YellowOutputCollection & out){}
  private:
    void do_something_specific_to_v2_firmware();
    YellowParams const & db;
    int data_specific_to_v2_firmware;
  };
  
} // namespace

#endif


