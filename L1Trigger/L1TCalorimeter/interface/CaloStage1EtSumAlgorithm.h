///
/// \class l1t::CaloStage1EtSumAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage1EtSumAlgorithm_h
#define CaloStage1EtSumAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <vector>

namespace l1t {
    
  class CaloStage1EtSumAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      std::vector<l1t::EtSum> & sums) = 0;    

    virtual ~CaloStage1EtSumAlgorithm(){};
  }; 
  
} 

#endif
