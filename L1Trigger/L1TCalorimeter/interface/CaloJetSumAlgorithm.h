///
/// \class l1t::CaloJetSumAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloJetSumAlgorithm_h
#define CaloJetSumAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/BXVector.h"
#include "DataFormats/L1TCalorimeter/interface/Jet.h"
#include "DataFormats/L1TCalorimeter/interface/EtSum.h"

#include "FWCore/Framework/interface/Event.h"

namespace l1t {
    
  class CaloJetSumAlgorithm { 
  public:
    virtual void processEvent(const BXVector<Jet> & jets,
							  BXVector<EtSum> & sums) = 0;    

    virtual ~CaloJetSumAlgorithm(){};
  }; 
  
} 

#endif