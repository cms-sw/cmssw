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

#ifndef CaloJetAlgorithm_h
#define CaloJetAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/BXVector.h"
#include "DataFormats/L1TCalorimeter/interface/Tower.h"
#include "DataFormats/L1TCalorimeter/interface/Jet.h"

#include "FWCore/Framework/interface/Event.h"

namespace l1t {
    
  class CaloJetAlgorithm { 
  public:
    virtual void processEvent(const BXVector<Tower> & towers,
							  BXVector<Jet> & jets) = 0;    

    virtual ~CaloJetAlgorithm(){};
  }; 
  
} 

#endif