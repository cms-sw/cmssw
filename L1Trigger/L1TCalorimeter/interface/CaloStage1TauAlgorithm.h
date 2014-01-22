///step03
/// \class l1t::CaloStage1TauAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage1TauAlgorithm_h
#define CaloStage1TauAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloStage1Cluster.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/L1Trigger/interface/Tau.h"

#include <vector>

namespace l1t {
    
  class CaloStage1TauAlgorithm { 
  public:
    virtual void processEvent(//const std::vector<l1t::CaloStage1Cluster> & clusters,
			      const std::vector<l1t::CaloEmCand> & clusters,	
			      const std::vector<l1t::CaloRegion> & regions,
			      std::vector<l1t::Tau> & taus) = 0;    

    virtual ~CaloStage1TauAlgorithm(){};
  }; 
  
} 

#endif
