///
/// \class l1t::Stage1Layer2HFRingSumAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: R. Alex Barbieri
///

//

#ifndef Stage1Layer2HFRingSumAlgorithm_h
#define Stage1Layer2HFRingSumAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include <vector>

namespace l1t {

  class Stage1Layer2HFRingSumAlgorithm {
  public:
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      const std::vector<l1t::Tau> * taus,
			      l1t::CaloSpare * spare) = 0;

    virtual ~Stage1Layer2HFRingSumAlgorithm(){};
  };

}

#endif
