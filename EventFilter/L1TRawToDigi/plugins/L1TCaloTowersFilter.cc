// -*- C++ -*-
//
// Package:    L1TValidationEventFilter
// Class:      L1TValidationEventFilter
//
/**\class L1TValidationEventFilter L1TValidationEventFilter.cc EventFilter/L1TRawToDigi/src/L1TValidationEventFilter.cc

Description: <one line class summary>
Implementation:
nn<Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <iostream>

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

//
// class declaration
//

namespace l1t {

  class L1TCaloTowersFilter : public edm::global::EDFilter<> {
  public:
    explicit L1TCaloTowersFilter(const edm::ParameterSet&);

  private:
    bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

    // ----------member data ---------------------------
    edm::EDGetTokenT<l1t::CaloTowerBxCollection> m_towerToken;

    int period_;  // validation event period
  };

  //
  // constructors and destructor
  //
  L1TCaloTowersFilter::L1TCaloTowersFilter(const edm::ParameterSet& iConfig)
      : period_(iConfig.getUntrackedParameter<int>("period", 107)) {
    //now do what ever initialization is needed

    edm::InputTag towerTag = iConfig.getParameter<edm::InputTag>("towerToken");
    m_towerToken = consumes<l1t::CaloTowerBxCollection>(towerTag);
  }

  //
  // member functions
  //

  // ------------ method called on each new Event  ------------
  bool L1TCaloTowersFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
    using namespace edm;

    Handle<BXVector<l1t::CaloTower> > towers;
    iEvent.getByToken(m_towerToken, towers);

    if (towers->size() == 0) {
      LogDebug("L1TCaloTowersFilter") << "Event does not contain towers." << std::endl;
      return false;
    }

    LogDebug("L1TCaloTowersFilter") << "Event does contains towers." << std::endl;
    return true;
  }

}  // namespace l1t

using namespace l1t;
//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloTowersFilter);
