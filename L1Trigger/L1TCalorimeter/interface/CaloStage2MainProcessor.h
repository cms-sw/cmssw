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


namespace l1t {
    
  class CaloStage2MainProcessor { 
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> &,
			      std::vector<l1t::EGamma> & egammas,
			      std::vector<l1t::Tau> & taus,
			      std::vector<l1t::Jet> & jets,
			      std::vector<l1t::EtSum> & etsums) = 0;    

    virtual ~CaloStage2MainProcessor(){};
  }; 
  
} 

#endif
