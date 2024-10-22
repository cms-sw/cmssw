// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1JetRecoTreeProducer
//
/**\class L1JetRecoTreeProducer L1JetRecoTreeProducer.cc L1Trigger/L1TNtuples/src/L1JetRecoTreeProducer.cc

 Description: Produces tree containing reco quantities


*/

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// photons
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoPhoton.h"

//
// class declaration
//

class L1PhotonRecoTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1PhotonRecoTreeProducer(const edm::ParameterSet&);
  ~L1PhotonRecoTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

public:
  L1Analysis::L1AnalysisRecoPhoton* photon;

  L1Analysis::L1AnalysisRecoPhotonDataFormat* photon_data;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags

  edm::EDGetTokenT<reco::PhotonCollection> PhotonToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > PhotonWP80MapToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > PhotonWP90MapToken_;

  // debug stuff
  bool photonsMissing_;
  unsigned int maxPhoton_;
};

L1PhotonRecoTreeProducer::L1PhotonRecoTreeProducer(const edm::ParameterSet& iConfig) : photonsMissing_(false) {
  maxPhoton_ = iConfig.getParameter<unsigned int>("maxPhoton");
  PhotonToken_ =
      consumes<reco::PhotonCollection>(iConfig.getUntrackedParameter("PhotonToken", edm::InputTag("photons")));

  PhotonWP80MapToken_ = consumes<edm::ValueMap<bool> >(
      iConfig.getUntrackedParameter("phoWP80MapToken", edm::InputTag("egmPhotonIDs:mvaPhoID-RunIIIWinter22-v1-wp80")));
  PhotonWP90MapToken_ = consumes<edm::ValueMap<bool> >(
      iConfig.getUntrackedParameter("phoWP90MapToken", edm::InputTag("egmPhotonIDs:mvaPhoID-RunIIIWinter22-v1-wp90")));

  photon = new L1Analysis::L1AnalysisRecoPhoton();
  photon_data = photon->getData();

  usesResource(TFileService::kSharedResource);
  tree_ = fs_->make<TTree>("PhotonRecoTree", "PhotonRecoTree");
  tree_->Branch("Photon", "L1Analysis::L1AnalysisRecoPhotonDataFormat", &photon_data, 32000, 3);
}

L1PhotonRecoTreeProducer::~L1PhotonRecoTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1PhotonRecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  photon->Reset();
  edm::Handle<reco::PhotonCollection> recoPhotons;
  iEvent.getByToken(PhotonToken_, recoPhotons);

  std::vector<edm::Handle<edm::ValueMap<bool> > > phoVIDDecisionHandles(2);

  iEvent.getByToken(PhotonWP80MapToken_, phoVIDDecisionHandles[0]);
  iEvent.getByToken(PhotonWP90MapToken_, phoVIDDecisionHandles[1]);

  if (recoPhotons.isValid() && phoVIDDecisionHandles[0].isValid() && phoVIDDecisionHandles[1].isValid()) {
    photon->SetPhoton(iEvent, iSetup, recoPhotons, phoVIDDecisionHandles, maxPhoton_);
  } else {
    if (!photonsMissing_) {
      edm::LogWarning("MissingProduct") << "Photons or photon ID not found.  Branch will not be filled" << std::endl;
    }
    photonsMissing_ = true;
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1PhotonRecoTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1PhotonRecoTreeProducer::endJob() {}

void L1PhotonRecoTreeProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("maxPhoton", 20);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1PhotonRecoTreeProducer);
