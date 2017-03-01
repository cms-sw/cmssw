///step03
/// \class l1t::Stage1Layer2TauAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage1Layer2TauAlgorithm_h
#define Stage1Layer2TauAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloStage1Cluster.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/L1Trigger/interface/Tau.h"

#include <vector>

namespace l1t {

  class Stage1Layer2TauAlgorithm {
  public:
    virtual void processEvent(//const std::vector<l1t::CaloStage1> & clusters,
			      const std::vector<l1t::CaloEmCand> & clusters,
			      const std::vector<l1t::CaloRegion> & regions,
			      std::vector<l1t::Tau> * isoTaus,
			      std::vector<l1t::Tau> * taus) = 0;

    virtual ~Stage1Layer2TauAlgorithm(){};
    std::string regionPUSType;
    std::vector<double> regionPUSParams;
  };

}

#endif
