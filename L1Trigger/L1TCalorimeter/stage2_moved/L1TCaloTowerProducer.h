///
/// \class l1t::CaloTowerProducer
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloTowerProducer_h
#define CaloTowerProducer_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "FWCore/Framework/interface/EDProducer.h"

namespace l1t {
    
  class L1TCaloTowerProducer : public edm::EDProducer { 
  public:
    virtual void processEvent(const EcalTriggerPrimitiveDigiCollection &,
							  const HcalTriggerPrimitiveCollection &,
 							  BXVector<l1t::CaloTower> & towers) = 0;    

    virtual ~L1TCaloTowerProducer(){};
  }; 
  
} 

#endif