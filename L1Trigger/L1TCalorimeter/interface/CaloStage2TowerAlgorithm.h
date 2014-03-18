///step03
/// \class l1t::CaloStage2TowerAlgorithm
///
/// Description: convert input quantities to ECAL + HCAL towers
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage2TowerAlgorithm_h
#define CaloStage2TowerAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"


namespace l1t {
    
  class CaloStage2TowerAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers) = 0;    

    virtual ~CaloStage2TowerAlgorithm(){};
  }; 
  
} 

#endif
