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

#ifndef CaloEtSumAlgorithm_h
#define CaloEtSumAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/BXVector.h"
#include "DataFormats/L1TCalorimeter/interface/Tower.h"
#include "DataFormats/L1TCalorimeter/interface/EtSum.h"

#include "FWCore/Framework/interface/Event.h"

namespace l1t {
    
  class CaloEtSumAlgorithm { 
  public:
    virtual void processEvent(const BXVector<Tower> & towers,
							  BXVector<EtSum> & sums) = 0;    

    virtual ~CaloEtSumAlgorithm(){};
  }; 
  
} 

#endif