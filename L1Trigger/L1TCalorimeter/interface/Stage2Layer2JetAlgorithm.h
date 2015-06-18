///
/// \class l1t::Stage2Layer2JetAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2JetAlgorithm_h
#define Stage2Layer2JetAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <vector>


namespace l1t {
    
  class Stage2Layer2JetAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & towers,
			      std::vector<l1t::Jet> & jets, std::vector<l1t::Jet> & alljets ) = 0;    

    virtual ~Stage2Layer2JetAlgorithm(){};

  }; 
  
} 

#endif
