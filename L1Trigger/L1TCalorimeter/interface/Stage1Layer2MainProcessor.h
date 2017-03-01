///
/// \class l1t::CaloMainProcessor
///
/// Description: interface for the main processor
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage1Layer2MainProcessor_h
#define Stage1Layer2MainProcessor_h


#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
//#include "DataFormats/L1Trigger/interface/BXVector.h"
#include <vector>
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"

#include "FWCore/Framework/interface/Event.h"

namespace l1t {

  class Stage1Layer2MainProcessor {
  public:
    virtual void processEvent(const std::vector<CaloEmCand> &,
			      const std::vector<CaloRegion> &,
			      std::vector<EGamma> * egammas,
			      std::vector<Tau> * taus,
			      std::vector<Tau> * isoTaus,
			      std::vector<Jet> * jets,
			      std::vector<Jet> * preGtJets,
			      std::vector<EtSum> * etsums,
			      CaloSpare * hfSums,
			      CaloSpare *hfCounts) = 0;

    virtual ~Stage1Layer2MainProcessor(){};
  };

}

#endif
