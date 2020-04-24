///step03
/// \class l1t::Stage2Layer2ClusterAlgorithm
///
/// Description: clustering algorithm for stage 2
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2ClusterAlgorithm_h
#define Stage2Layer2ClusterAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

namespace l1t {
    
  class Stage2Layer2ClusterAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::CaloCluster> & clusters) = 0;    

    virtual ~Stage2Layer2ClusterAlgorithm(){};
  }; 
  
} 

#endif
