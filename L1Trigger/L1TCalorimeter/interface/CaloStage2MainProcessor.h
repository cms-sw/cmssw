///
/// \class l1t::CaloStage2MainProcessor
///
/// Description: interface for the main processor
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef CaloStage2MainProcessor_h
#define CaloStage2MainProcessor_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"


namespace l1t {
    
  class CaloStage2MainProcessor { 
  public:
    virtual void processEvent(const BXVector<l1t::CaloTower> &,
 							  BXVector<l1t::EGamma> & egammas,
							  BXVector<l1t::Tau> & taus,
							  BXVector<l1t::Jet> & jets,
							  BXVector<l1t::EtSum> & etsums) = 0;    

    virtual ~CaloMainProcessor(){};
  }; 
  
} 

#endif