///
/// \class l1t::Stage1Layer2FirmwareFactory
///
/// Implementation:
/// Demonstrates how to define the firmware interface.
///
/// \author: R. Alex Barbieri MIT
///


#ifndef CALOSTAGE1JETALGORITHMFACTORY_H
#define CALOSTAGE1JETALGORITHMFACTORY_H

#include <boost/shared_ptr.hpp>

#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2MainProcessor.h"

namespace l1t {

  class Stage1Layer2FirmwareFactory {
  public:
    typedef boost::shared_ptr<Stage1Layer2MainProcessor> ReturnType;

    // ReturnType create(const FirmwareVersion & fwv /*,const CaloParamsHelper & dbPars*/);
    ReturnType create(const int fwv ,CaloParamsHelper* dbPars);

    // (Why not make "create" a static member function? You could...
    // But this way allows you to add additional customizations to the
    // factory not necessarily coming from the DB.)
  };

} // namespace

#endif
///
/// \class l1t::
///
/// Description: fictitious Yellow trigger.
///
/// Implementation:
/// Demonstrates how to
///
/// \author: Michael Mulhearn - UC Davis
///
