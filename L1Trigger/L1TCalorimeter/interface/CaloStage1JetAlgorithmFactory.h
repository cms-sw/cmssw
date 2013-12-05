///
/// \class l1t::CaloStage1JetAlgorithm
///
/// Implementation:
/// Demonstrates how to define the firmware interface.
///
/// \author: R. Alex Barbieri MIT
///

//
// See CaloStage1JetAlgorithm.h for a complete description of the recommended Firmware design pattern.
//

#ifndef CALOSTAGE1JETALGORITHMFACTORY_H
#define CALOSTAGE1JETALGORITHMFACTORY_H

#include <boost/shared_ptr.hpp>

#include "CondFormats/L1TYellow/interface/CaloParams.h"
//#include "DataFormats/L1TYellow/interface/YellowDigi.h"
//#include "DataFormats/L1TYellow/interface/YellowOutput.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/L1TYellow/interface/CaloStage1JetAlgorithm.h"

namespace l1t {

  class CaloStage1JetAlgorithmFactory {
  public:
    typedef boost::shared_ptr<CaloStage1JetAlgorithm> ReturnType;

    ReturnType create(const CaloParams & dbPars);

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
