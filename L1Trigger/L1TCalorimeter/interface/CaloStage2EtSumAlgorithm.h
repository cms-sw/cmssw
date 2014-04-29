///
/// \class l1t::CaloStage2JetSumAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage2EtSumAlgorithm_h
#define CaloStage2EtSumAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <vector>

namespace l1t {
    
  class CaloStage2EtSumAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::EtSum> & sums) = 0;    

    virtual ~CaloStage2EtSumAlgorithm(){};
  }; 
  
} 

#endif
