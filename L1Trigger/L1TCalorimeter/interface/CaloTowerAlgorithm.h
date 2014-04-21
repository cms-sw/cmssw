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

#ifndef CaloTowerAlgorithm_h
#define CaloTowerAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/BXVector.h"
#include "DataFormats/L1TCalorimeter/interface/EGamma.h"
#include "DataFormats/L1TCalorimeter/interface/Tau.h"
#include "DataFormats/L1TCalorimeter/interface/Jet.h"
#include "DataFormats/L1TCalorimeter/interface/EtSum.h"

#include "FWCore/Framework/interface/Event.h"

namespace l1t {
    
  class CaloTowerAlgorithm { 
  public:
    virtual void processEvent(const EcalTriggerPrimitiveDigiCollection &,
							  const HcalTriggerPrimitiveCollection &,
 							  BXVector<Tower> & towers) = 0;    

    virtual ~CaloTowerAlgorithm(){};
  }; 
  
} 

#endif