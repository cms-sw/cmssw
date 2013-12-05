///
/// \class l1t::CaloStage2JetAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage1JetAlgorithm_h
#define CaloStage1JetAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class CaloStage1JetAlgorithm {
  public:
    virtual void processEvent(const BXVector<l1t::CaloRegion> & regions,
			      BXVector<l1t::Jet> & jets) = 0;

    virtual ~CaloStage1JetAlgorithm(){};
  };

}

#endif
