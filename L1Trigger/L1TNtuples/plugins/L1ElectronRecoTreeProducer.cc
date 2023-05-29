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

//electrons
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoElectron.h"

//
// class declaration
//

class L1ElectronRecoTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1ElectronRecoTreeProducer(const edm::ParameterSet&);
  ~L1ElectronRecoTreeProducer() override;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

public:
  L1Analysis::L1AnalysisRecoElectron* electron;

  L1Analysis::L1AnalysisRecoElectronDataFormat* electron_data;

private:
  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  //edm::EDGetToken ElectronToken_;
  // edm::EDGetToken<edm::View<reco::GsfElectron>> ElectronToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> ElectronToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > ElectronVetoIdMapToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > ElectronLooseIdMapToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > ElectronMediumIdMapToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> > ElectronTightIdMapToken_;

  // debug stuff
  bool electronsMissing_;
  unsigned int maxElectron_;
};

L1ElectronRecoTreeProducer::L1ElectronRecoTreeProducer(const edm::ParameterSet& iConfig) : electronsMissing_(false) {
  maxElectron_ = iConfig.getParameter<unsigned int>("maxElectron");
  //ElectronToken_ = mayConsume<edm::View<reco::GsfElectron> >(iConfig.getUntrackedParameter("ElectronToken",edm::InputTag("gedGsfElectrons")));
  ElectronToken_ = consumes<reco::GsfElectronCollection>(
      iConfig.getUntrackedParameter("ElectronToken", edm::InputTag("gedGsfElectrons")));

  /*RhoToken_ = consumes<double>(iConfig.getUntrackedParameter("RhoToken",edm::InputTag("fixedGridRhoFastjetAllCalo")));
  vtxToken_          = mayConsume<reco::VertexCollection>(iConfig.getUntrackedParameter("vtxToken",edm::InputTag("offlinePrimaryVertices")));
  conversionsToken_ = mayConsume< reco::ConversionCollection >(iConfig.getUntrackedParameter("conversionsToken",edm::InputTag("conversions")));
  beamSpotToken_ = mayConsume< reco::BeamSpot>(iConfig.getUntrackedParameter("beamSpotToken",edm::InputTag("offlineBeamSpot")));*/

  ElectronVetoIdMapToken_ = consumes<edm::ValueMap<bool> >(iConfig.getUntrackedParameter(
      "eleVetoIdMapToken", edm::InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto")));
  ElectronLooseIdMapToken_ = consumes<edm::ValueMap<bool> >(iConfig.getUntrackedParameter(
      "eleLooseIdMapToken", edm::InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-loose")));
  ElectronMediumIdMapToken_ = consumes<edm::ValueMap<bool> >(iConfig.getUntrackedParameter(
      "eleMediumIdMapToken", edm::InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-medium")));
  ElectronTightIdMapToken_ = consumes<edm::ValueMap<bool> >(iConfig.getUntrackedParameter(
      "eleTightIdMapToken", edm::InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-tight")));

  electron = new L1Analysis::L1AnalysisRecoElectron();
  electron_data = electron->getData();

  usesResource(TFileService::kSharedResource);
  tree_ = fs_->make<TTree>("ElectronRecoTree", "ElectronRecoTree");
  tree_->Branch("Electron", "L1Analysis::L1AnalysisRecoElectronDataFormat", &electron_data, 32000, 3);
}

L1ElectronRecoTreeProducer::~L1ElectronRecoTreeProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1ElectronRecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  electron->Reset();
  //edm::Handle<edm::View<reco::GsfElectron> > recoElectrons;
  //iEvent.getByToken(ElectronToken_, recoElectrons);
  edm::Handle<reco::GsfElectronCollection> recoElectrons;
  iEvent.getByToken(ElectronToken_, recoElectrons);

  std::vector<edm::Handle<edm::ValueMap<bool> > > eleVIDDecisionHandles(4);

  iEvent.getByToken(ElectronVetoIdMapToken_, eleVIDDecisionHandles[0]);
  iEvent.getByToken(ElectronLooseIdMapToken_, eleVIDDecisionHandles[1]);
  iEvent.getByToken(ElectronMediumIdMapToken_, eleVIDDecisionHandles[2]);
  iEvent.getByToken(ElectronTightIdMapToken_, eleVIDDecisionHandles[3]);

  if (recoElectrons.isValid() && eleVIDDecisionHandles[0].isValid() && eleVIDDecisionHandles[1].isValid() &&
      eleVIDDecisionHandles[2].isValid() && eleVIDDecisionHandles[3].isValid()) {
    electron->SetElectron(iEvent, iSetup, recoElectrons, eleVIDDecisionHandles, maxElectron_);
  } else {
    if (!electronsMissing_) {
      edm::LogWarning("MissingProduct") << "CaloJets not found.  Branch will not be filled" << std::endl;
    }
    electronsMissing_ = true;
  }

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1ElectronRecoTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1ElectronRecoTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1ElectronRecoTreeProducer);
