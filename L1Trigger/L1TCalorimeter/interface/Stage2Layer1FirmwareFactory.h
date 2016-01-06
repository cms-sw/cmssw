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

#include <boost/shared_ptr.hpp>

#include "L1Trigger/L1TCalorimeter/interface/Stage2PreProcessor.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"


namespace l1t {

  class Stage2Layer1FirmwareFactory {
  public:
    typedef boost::shared_ptr<Stage2PreProcessor> ReturnType;

    ReturnType create(unsigned fwv, CaloParamsHelper* params);

  };

} // namespace

#endif
