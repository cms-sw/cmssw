///step03
/// \class l1t::Stage2TowerDecompressAlgorithm
///
/// Description: convert input quantities to ECAL + HCAL towers
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2TowerDecompressAlgorithm_h
#define Stage2TowerDecompressAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"


namespace l1t {
    
  class Stage2TowerDecompressAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers) = 0;    

    virtual ~Stage2TowerDecompressAlgorithm(){};
  }; 
  
} 

#endif
