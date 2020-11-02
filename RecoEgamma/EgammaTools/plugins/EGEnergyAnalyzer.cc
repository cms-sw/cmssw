// -*- C++ -*-
//
// Package:    EGEnergyAnalyzer
// Class:      EGEnergyAnalyzer
//
/**\class EGEnergyAnalyzer EGEnergyAnalyzer.cc GBRWrap/EGEnergyAnalyzer/src/EGEnergyAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Josh Bendavid
//         Created:  Tue Nov  8 22:26:45 CET 2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TFile.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergyCorrector.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

//
// class declaration
//

class EGEnergyAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EGEnergyAnalyzer(const edm::ParameterSet&);
  ~EGEnergyAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  EGEnergyCorrector corfile;
  EGEnergyCorrector cordb;

  edm::EDGetTokenT<EcalRecHitCollection> ebRHToken_, eeRHToken_;
  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;
};

EGEnergyAnalyzer::EGEnergyAnalyzer(const edm::ParameterSet& iConfig)
    : ecalClusterToolsESGetTokens_{consumesCollector()} {
  ebRHToken_ = consumes(edm::InputTag("reducedEcalRecHitsEB"));
  eeRHToken_ = consumes(edm::InputTag("reducedEcalRecHitsEE"));
}

EGEnergyAnalyzer::~EGEnergyAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void EGEnergyAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (!corfile.IsInitialized()) {
    corfile.Initialize(iSetup, "/afs/cern.ch/user/b/bendavid/cmspublic/gbrv3ph.root");
    //corfile.Initialize(iSetup,"wgbrph",true);
  }

  if (!cordb.IsInitialized()) {
    //cordb.Initialize(iSetup,"/afs/cern.ch/user/b/bendavid/cmspublic/regweights/gbrph.root");
    cordb.Initialize(iSetup, "wgbrph", true);
  }

  // get photon collection
  Handle<reco::PhotonCollection> hPhotonProduct;
  iEvent.getByLabel("photons", hPhotonProduct);

  EcalClusterLazyTools lazyTools(iEvent, ecalClusterToolsESGetTokens_.get(iSetup), ebRHToken_, eeRHToken_);

  Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByLabel("offlinePrimaryVerticesWithBS", hVertexProduct);

  for (reco::PhotonCollection::const_iterator it = hPhotonProduct->begin(); it != hPhotonProduct->end(); ++it) {
    std::pair<double, double> corsfile = corfile.CorrectedEnergyWithError(*it, *hVertexProduct, lazyTools, iSetup);
    std::pair<double, double> corsdb = cordb.CorrectedEnergyWithError(*it, *hVertexProduct, lazyTools, iSetup);

    printf("file: default = %5f, correction = %5f, uncertainty = %5f\n", it->energy(), corsfile.first, corsfile.second);
    printf("db:   default = %5f, correction = %5f, uncertainty = %5f\n", it->energy(), corsdb.first, corsdb.second);
  }
}

// ------------ method called once each job just before starting event loop  ------------
void EGEnergyAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void EGEnergyAnalyzer::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EGEnergyAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EGEnergyAnalyzer);
