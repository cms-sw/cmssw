///
/// \class l1t::CaloStage2JetAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage2JetAlgorithm_h
#define CaloStage2JetAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/Jet.h"

#include <vector>


namespace l1t {
    
  class CaloStage2JetAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::Jet> & jets) = 0;    

    virtual ~CaloStage2JetAlgorithm(){};
  }; 
  
} 

#endif
