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

#ifndef CaloStage2JetSumAlgorithm_h
#define CaloStage2JetSumAlgorithm_h

#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
    
  class CaloStage2JetSumAlgorithm { 
  public:
    virtual void processEvent(const BXVector<l1t::Jet> & jets,
							  BXVector<l1t::EtSum> & sums) = 0;    

    virtual ~CaloStage2JetSumAlgorithm(){};
  }; 
  
} 

#endif