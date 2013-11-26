///step03
/// \class l1t::CaloStage1EGammaAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage1EGammaAlgorithm_h
#define CaloStage1EGammaAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloStage1Cluster.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
    
  class CaloStage1EGammaAlgorithm { 
  public:
    virtual void processEvent(const BXVector<l1t::CaloStage1Cluster> & clusters,
							  const BXVector<l1t::CaloRegion> & regions,
							  BXVector<l1t::EGamma> & egammas) = 0;    

    virtual ~CaloStage1EGammaAlgorithm(){};
  }; 
  
} 

#endif