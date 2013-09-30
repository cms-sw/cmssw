#ifndef L1TYellowFirmware_h
#define L1TYellowFirmware_h

#include "L1Trigger/L1TYellow/interface/L1TYellowAlg.h"

//
//  L1TYellowFirmware:
//
//    Class declarations for entire collection of version dependent firmware.
//
//    All Firmware versions satisfy the L1TYellowAlg interface.
//

namespace l1t {
  
  class L1TYellowAlg_v1 : public L1TYellowAlg {
  public:
    L1TYellowAlg_v1(const L1TYellowParams & dbPars);
    virtual ~L1TYellowAlg_v1();
    virtual void processEvent(const L1TYellowDigiCollection & input, L1TYellowOutputCollection & out);
  private:
    void do_something_specific_to_v1_firmware();
    L1TYellowParams const & db;
    int data_specific_to_v1_firmware;
  };

  class L1TYellowAlg_v2 : public L1TYellowAlg {
  public:
    L1TYellowAlg_v2(const L1TYellowParams & dbPars);
    virtual ~L1TYellowAlg_v2(){};
    virtual void processEvent(const L1TYellowDigiCollection & input, L1TYellowOutputCollection & out){}
  private:
    void do_something_specific_to_v2_firmware();
    L1TYellowParams const & db;
    int data_specific_to_v2_firmware;
  };
  
} // namespace

#endif


