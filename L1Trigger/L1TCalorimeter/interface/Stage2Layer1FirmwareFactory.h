///
/// \class l1t::CaloStage1FirmwareFactory
///
/// Implementation:
/// Demonstrates how to define the firmware interface.
///
/// \author: R. Alex Barbieri MIT
///

#ifndef Stage2Layer1FirmwareFactory_h
#define Stage2Layer1FirmwareFactory_h

#include <memory>

#include "L1Trigger/L1TCalorimeter/interface/Stage2PreProcessor.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

namespace l1t {

  class Stage2Layer1FirmwareFactory {
  public:
    typedef std::unique_ptr<Stage2PreProcessor> ReturnType;

    ReturnType create(unsigned fwv, CaloParamsHelper const* params);
  };

}  // namespace l1t

#endif
