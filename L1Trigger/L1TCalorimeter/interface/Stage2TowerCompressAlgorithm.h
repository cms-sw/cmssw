///step03
/// \class l1t::Stage2TowerCompressAlgorithm
///
/// Description: compress towers into sum+ratio format
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2TowerCompressAlgorithm_h
#define Stage2TowerCompressAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"


namespace l1t {
    
  class Stage2TowerCompressAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers) = 0;    

    virtual ~Stage2TowerCompressAlgorithm(){};
  }; 
  
} 

#endif
