///
/// \class l1t::CaloStage1FirmwareFactory
///
/// Implementation:
/// Demonstrates how to define the firmware interface.
///
/// \author: R. Alex Barbieri MIT
///


#ifndef CALOSTAGE2FIRMWAREFACTORY_H
#define CALOSTAGE2FIRMWAREFACTORY_H

#include <boost/shared_ptr.hpp>

//#include "CondFormats/L1TCalorimeter/interface/CaloParams.h"
#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloStage2MainProcessor.h"

namespace l1t {

  class CaloStage2FirmwareFactory {
  public:
    typedef boost::shared_ptr<CaloStage2MainProcessor> ReturnType;

    ReturnType create(const FirmwareVersion & fwv);

    // (Why not make "create" a static member function? You could...
    // But this way allows you to add additional customizations to the
    // factory not necessarily coming from the DB.)
  };

} // namespace

#endif
