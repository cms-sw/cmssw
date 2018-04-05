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

#ifndef Stage2Layer2JetSumAlgorithm_h
#define Stage2Layer2JetSumAlgorithm_h

#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <vector>

namespace l1t {
    
  class Stage2Layer2JetSumAlgorithm { 
  public:
    virtual void processEvent(const std::vector<l1t::Jet> & alljets,
			      std::vector<l1t::EtSum> & htsums) = 0;    

    virtual ~Stage2Layer2JetSumAlgorithm(){};
  }; 
  
} 

#endif
