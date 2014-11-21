///
/// \class l1t::Stage2Layer2TauAlgorithm
///
/// Description: Tau algorithm interface, separate clustering
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2TauAlgorithm_h
#define Stage2Layer2TauAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/Tau.h"

#include <vector>


namespace l1t {
    
  class Stage2Layer2TauAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloCluster> & clusters,
    						  const std::vector<l1t::CaloTower>& towers,
			      			  std::vector<l1t::Tau> & taus) = 0;    

    virtual ~Stage2Layer2TauAlgorithm(){};
  }; 
  
} 

#endif
