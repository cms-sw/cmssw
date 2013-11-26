///
/// \class l1t::CaloMainProcessorFirmware
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloClusterAlgorithm_h
#define CaloClusterAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/BXVector.h"
#include "DataFormats/L1TCalorimeter/interface/EGamma.h"
#include "DataFormats/L1TCalorimeter/interface/Tau.h"
#include "DataFormats/L1TCalorimeter/interface/Jet.h"
#include "DataFormats/L1TCalorimeter/interface/EtSum.h"

#include "FWCore/Framework/interface/Event.h"

namespace l1t {
    
  class CaloClusterAlgorithm { 
  public:
    virtual void processEvent(const BXVector<Tower> & towers,
							  BXVector<Cluster> & clusters) = 0;    

    virtual ~CaloClusterAlgorithm(){};
  }; 
  
} 

#endif