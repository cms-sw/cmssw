///
/// \class l1t::CaloStage2TauAlgorithm
///
/// Description: Tau algorithm interface, separate clustering
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage2TauAlgorithm_h
#define CaloStage2TauAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

#include "DataFormats/L1Trigger/interface/Tau.h"

#include <vector>


namespace l1t {
    
  class CaloStage2TauAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloCluster> & clusters,
			      std::vector<l1t::Tau> & taus) = 0;    

    virtual ~CaloStage2TauAlgorithm(){};
  }; 
  
} 

#endif
