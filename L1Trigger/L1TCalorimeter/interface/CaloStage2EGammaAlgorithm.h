///
/// \class l1t::CaloStage2EGammaAlgorithm
///
/// Description: EG algorithm interface, separate clustering
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage2EGammaAlgorithm_h
#define CaloStage2EGammaAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include <vector>


namespace l1t {
    
  class CaloStage2EGammaAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloCluster> & clusters,
			      std::vector<l1t::EGamma> & egammas) = 0;    

    virtual ~CaloStage2EGammaAlgorithm(){};
  }; 
  
} 

#endif
