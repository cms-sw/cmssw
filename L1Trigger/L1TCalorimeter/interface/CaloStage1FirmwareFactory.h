///
/// \class l1t::CaloStage1FirmwareFactory
///
/// Implementation:
/// Demonstrates how to define the firmware interface.
///
/// \author: R. Alex Barbieri MIT
///


#ifndef CALOSTAGE1JETALGORITHMFACTORY_H
#define CALOSTAGE1JETALGORITHMFACTORY_H

#include <boost/shared_ptr.hpp>

//#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1MainProcessor.h"

namespace l1t {

  class CaloStage1FirmwareFactory {
  public:
    typedef boost::shared_ptr<CaloStage1MainProcessor> ReturnType;

    ReturnType create(const FirmwareVersion & fwv /*,const CaloParams & dbPars*/);

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
