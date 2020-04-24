///
/// \class l1t::Stage1Layer2EtSumAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage1Layer2EtSumAlgorithm_h
#define Stage1Layer2EtSumAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include <vector>

namespace l1t {

  class Stage1Layer2EtSumAlgorithm {
  public:
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      const std::vector<l1t::Jet> * jets,
			      std::vector<l1t::EtSum> * sums) = 0;

    virtual ~Stage1Layer2EtSumAlgorithm(){};
  };

}

#endif
