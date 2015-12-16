///
/// \class l1t::CaloStage1FirmwareFactory
///
/// Implementation:
/// Demonstrates how to define the firmware interface.
///
/// \author: R. Alex Barbieri MIT
///


#ifndef Stage2Layer2FirmwareFactory_h
#define Stage2Layer2FirmwareFactory_h

#include <boost/shared_ptr.hpp>

#include "L1Trigger/L1TCalorimeter/interface/Stage2MainProcessor.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

//#include "FWCore/Framework/interface/Event.h"


namespace l1t {

  class Stage2Layer2FirmwareFactory {
  public:
    typedef boost::shared_ptr<Stage2MainProcessor> ReturnType;

    ReturnType create(unsigned fwv, CaloParamsHelper* params);

  };

} // namespace

#endif
