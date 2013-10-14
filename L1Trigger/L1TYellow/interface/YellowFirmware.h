///
/// \class l1t::YellowFirmware
///
/// Description: Firmware interface for the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to define the firmware interface.
///
/// \author: Michael Mulhearn - UC Davis
///

//
//  The recommended design pattern for emulating system firmware is implemented
//  in the following files:
// 
//   L1Trigger/L1TYellow/interface/YellowFirmwareFactory.h
//   L1Trigger/L1TYellow/interface/YellowFirmware.h
//   L1Trigger/L1TYellow/src/firmware/YellowFirmwareImp.h
//   L1Trigger/L1TYellow/src/firmware/YellowFirmwareFactory.cc
//   L1Trigger/L1TYellow/src/firmware/YellowFirmwareImp1.cc
//   L1Trigger/L1TYellow/src/firmware/YellowFirmwareImp2.cc
// 
//  This class (YellowFirmware) is an abstract base class.  It defines the
//  interface used by all concrete implementations of the firmware.  If future
//  firmware requires additional inputs or outputs, the interface can be
//  extended as needed.
//
//  At the beginning of each run, a check is made as to whether the
//  configuration parameters for the trigger have been updated.  If so, they are
//  updated, and the firmware factory class (YellowFirmwareFactory) is used to
//  create a new  firmware instance corresponding to these new parameters.
//
//  Have a look at YellowFirmwareFactory.cc and the concrete implementations of
//  the firmware.  Note that, given a new firmware version, it is completely up
//  to the developer as to whether the new firmware should be emulated in a
//  whole new implementation (probably best for major changes) or by using
//  version switching in an existing implementation (probably best for minor
//  changes).
//

#ifndef YELLOWFIRMWARE_H
#define YELLOWFIRMWARE_H

#include "CondFormats/L1TYellow/interface/YellowParams.h"
#include "DataFormats/L1TYellow/interface/YellowDigi.h"
#include "DataFormats/L1TYellow/interface/YellowOutput.h"
#include "FWCore/Framework/interface/Event.h"

namespace l1t {
    
  class YellowFirmware { 
  public:
    virtual void processEvent(const YellowDigiCollection &, YellowOutputCollection & out) = 0;    
    virtual ~YellowFirmware(){};
  }; 
  
} 

#endif
