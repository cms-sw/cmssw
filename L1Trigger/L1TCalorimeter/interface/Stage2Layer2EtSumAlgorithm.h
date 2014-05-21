///
/// \class l1t::Stage2Layer2JetSumAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2EtSumAlgorithm_h
#define Stage2Layer2EtSumAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <vector>

namespace l1t {
    
  class Stage2Layer2EtSumAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::EtSum> & sums) = 0;    

    virtual ~Stage2Layer2EtSumAlgorithm(){};
  }; 
  
} 

#endif
