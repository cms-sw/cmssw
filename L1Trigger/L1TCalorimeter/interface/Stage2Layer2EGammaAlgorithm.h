///
/// \class l1t::Stage2Layer2EGammaAlgorithm
///
/// Description: EG algorithm interface, separate clustering
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2EGammaAlgorithm_h
#define Stage2Layer2EGammaAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"


#include <vector>


namespace l1t {
    
  class Stage2Layer2EGammaAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloCluster> & clusters, 
			      const std::vector<CaloTower>& towers,
			      std::vector<l1t::EGamma> & egammas) = 0;    

    virtual ~Stage2Layer2EGammaAlgorithm(){};
  }; 
  
} 

#endif
