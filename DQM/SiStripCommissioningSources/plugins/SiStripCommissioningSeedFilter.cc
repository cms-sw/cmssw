// -*- C++ -*-
//
// Package:    SiStripCommissioningSeedFilter
// Class:      SiStripCommissioningSeedFilter
//
/**\class SiStripCommissioningSeedFilter SiStripCommissioningSeedFilter.cc myTestArea/SiStripCommissioningSeedFilter/src/SiStripCommissioningSeedFilter.cc

 Description: simply filter acording to the run type

 Implementation:
     Uses information from SiStripEventSummary, so it has to be called after Raw2Digi.
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Fri Jan 18 12:17:46 CET 2008
//
//

// system include files
#include <memory>
#include <algorithm>
#include <vector>

// user include files
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class SiStripCommissioningSeedFilter : public edm::stream::EDFilter<> {
public:
  explicit SiStripCommissioningSeedFilter(const edm::ParameterSet&);
  ~SiStripCommissioningSeedFilter() override = default;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  //      edm::InputTag inputModuleLabel_;
  edm::EDGetTokenT<TrajectorySeedCollection> seedcollToken_;
};

//
// constructors and destructor
//
SiStripCommissioningSeedFilter::SiStripCommissioningSeedFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  //   inputModuleLabel_ = iConfig.getParameter<edm::InputTag>( "InputModuleLabel" ) ;
  seedcollToken_ = consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("InputModuleLabel"));
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool SiStripCommissioningSeedFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<TrajectorySeedCollection> seedcoll;
  iEvent.getByToken(seedcollToken_, seedcoll);
  bool result = !(*seedcoll).empty();
  return result;
}

DEFINE_FWK_MODULE(SiStripCommissioningSeedFilter);
