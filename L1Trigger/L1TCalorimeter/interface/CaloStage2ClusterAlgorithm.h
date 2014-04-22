///step03
/// \class l1t::CaloStage2MainProcessorFirmware
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage2ClusterAlgorithm_h
#define CaloStage2ClusterAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

namespace l1t {
    
  class CaloStage2ClusterAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::CaloCluster> & clusters) = 0;    

    virtual ~CaloStage2ClusterAlgorithm(){};
  }; 
  
} 

#endif
