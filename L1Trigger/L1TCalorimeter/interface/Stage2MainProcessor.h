///
/// \class l1t::Stage2MainProcessor
///
/// Description: interface for the main processor
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2MainProcessor_h
#define Stage2MainProcessor_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

namespace l1t {

  class Stage2MainProcessor {
  public:
    virtual void processEvent(const std::vector<l1t::CaloTower> & inTowers,
			      std::vector<l1t::CaloTower> & outTowers,
			      std::vector<l1t::CaloCluster> & clusters,
			      std::vector<l1t::EGamma> & mpEGammas,
			      std::vector<l1t::Tau> & mpTaus,
			      std::vector<l1t::Jet> & mpJets,
			      std::vector<l1t::EtSum> & mpSums,
			      std::vector<l1t::EGamma> & egammas,
			      std::vector<l1t::Tau> & taus,
			      std::vector<l1t::Jet> & jets,
			      std::vector<l1t::EtSum> & etSums) = 0;

    virtual ~Stage2MainProcessor(){};

  };

}

#endif
