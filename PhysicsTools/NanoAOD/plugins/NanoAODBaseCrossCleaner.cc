// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      NanoAODBaseCrossCleaner
//
/**\class NanoAODBaseCrossCleaner NanoAODBaseCrossCleaner.cc PhysicsTools/NanoAODBaseCrossCleaner/plugins/NanoAODBaseCrossCleaner.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Mon, 28 Aug 2017 09:26:39 GMT
//
//

#include "PhysicsTools/NanoAOD/plugins/NanoAODBaseCrossCleaner.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

//
// constructors and destructor
//
NanoAODBaseCrossCleaner::NanoAODBaseCrossCleaner(const edm::ParameterSet& params)
    : name_(params.getParameter<std::string>("name")),
      doc_(params.getParameter<std::string>("doc")),
      jets_(consumes<edm::View<pat::Jet>>(params.getParameter<edm::InputTag>("jets"))),
      muons_(consumes<edm::View<pat::Muon>>(params.getParameter<edm::InputTag>("muons"))),
      electrons_(consumes<edm::View<pat::Electron>>(params.getParameter<edm::InputTag>("electrons"))),
      lowPtElectronsTag_(params.getParameter<edm::InputTag>("lowPtElectrons")),
      lowPtElectrons_(mayConsume<edm::View<pat::Electron>>(lowPtElectronsTag_)),
      taus_(consumes<edm::View<pat::Tau>>(params.getParameter<edm::InputTag>("taus"))),
      photons_(consumes<edm::View<pat::Photon>>(params.getParameter<edm::InputTag>("photons"))),
      jetSel_(params.getParameter<std::string>("jetSel")),
      muonSel_(params.getParameter<std::string>("muonSel")),
      electronSel_(params.getParameter<std::string>("electronSel")),
      lowPtElectronSel_(params.getParameter<std::string>("lowPtElectronSel")),
      tauSel_(params.getParameter<std::string>("tauSel")),
      photonSel_(params.getParameter<std::string>("photonSel")),
      jetName_(params.getParameter<std::string>("jetName")),
      muonName_(params.getParameter<std::string>("muonName")),
      electronName_(params.getParameter<std::string>("electronName")),
      lowPtElectronName_(params.getParameter<std::string>("lowPtElectronName")),
      tauName_(params.getParameter<std::string>("tauName")),
      photonName_(params.getParameter<std::string>("photonName"))

{
  produces<nanoaod::FlatTable>("jets");
  produces<nanoaod::FlatTable>("muons");
  produces<nanoaod::FlatTable>("electrons");
  if (!lowPtElectronsTag_.label().empty())
    produces<nanoaod::FlatTable>("lowPtElectrons");
  produces<nanoaod::FlatTable>("taus");
  produces<nanoaod::FlatTable>("photons");
}

NanoAODBaseCrossCleaner::~NanoAODBaseCrossCleaner() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void NanoAODBaseCrossCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::Handle<edm::View<pat::Jet>> jetsIn;
  iEvent.getByToken(jets_, jetsIn);
  std::vector<uint8_t> jets;
  for (const auto& j : *jetsIn) {
    jets.push_back(jetSel_(j));
  }
  auto jetsTable = std::make_unique<nanoaod::FlatTable>(jetsIn->size(), jetName_, false, true);

  edm::Handle<edm::View<pat::Muon>> muonsIn;
  iEvent.getByToken(muons_, muonsIn);
  std::vector<uint8_t> muons;
  for (const auto& m : *muonsIn) {
    muons.push_back(muonSel_(m));
  }
  auto muonsTable = std::make_unique<nanoaod::FlatTable>(muonsIn->size(), muonName_, false, true);

  edm::Handle<edm::View<pat::Electron>> electronsIn;
  iEvent.getByToken(electrons_, electronsIn);
  std::vector<uint8_t> eles;
  for (const auto& e : *electronsIn) {
    eles.push_back(electronSel_(e));
  }
  auto electronsTable = std::make_unique<nanoaod::FlatTable>(electronsIn->size(), electronName_, false, true);

  edm::Handle<edm::View<pat::Electron>> lowPtElectronsIn;
  std::vector<uint8_t> lowPtEles;
  if (!lowPtElectronsTag_.label().empty()) {
    iEvent.getByToken(lowPtElectrons_, lowPtElectronsIn);
    for (const auto& e : *lowPtElectronsIn) {
      lowPtEles.push_back(lowPtElectronSel_(e));
    }
  }

  edm::Handle<edm::View<pat::Tau>> tausIn;
  iEvent.getByToken(taus_, tausIn);
  std::vector<uint8_t> taus;
  for (const auto& t : *tausIn) {
    taus.push_back(tauSel_(t));
  }
  auto tausTable = std::make_unique<nanoaod::FlatTable>(tausIn->size(), tauName_, false, true);

  edm::Handle<edm::View<pat::Photon>> photonsIn;
  iEvent.getByToken(photons_, photonsIn);
  std::vector<uint8_t> photons;
  for (const auto& p : *photonsIn) {
    photons.push_back(photonSel_(p));
  }
  auto photonsTable = std::make_unique<nanoaod::FlatTable>(photonsIn->size(), photonName_, false, true);

  objectSelection(*jetsIn, *muonsIn, *electronsIn, *tausIn, *photonsIn, jets, muons, eles, taus, photons);

  muonsTable->addColumn<uint8_t>(name_, muons, doc_);
  jetsTable->addColumn<uint8_t>(name_, jets, doc_);
  electronsTable->addColumn<uint8_t>(name_, eles, doc_);
  tausTable->addColumn<uint8_t>(name_, taus, doc_);
  photonsTable->addColumn<uint8_t>(name_, photons, doc_);

  iEvent.put(std::move(jetsTable), "jets");
  iEvent.put(std::move(muonsTable), "muons");
  iEvent.put(std::move(electronsTable), "electrons");
  iEvent.put(std::move(tausTable), "taus");
  iEvent.put(std::move(photonsTable), "photons");

  if (!lowPtElectronsTag_.label().empty()) {
    auto lowPtElectronsTable =
        std::make_unique<nanoaod::FlatTable>(lowPtElectronsIn->size(), lowPtElectronName_, false, true);
    lowPtElectronsTable->addColumn<uint8_t>(name_, lowPtEles, doc_);
    iEvent.put(std::move(lowPtElectronsTable), "lowPtElectrons");
  }
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void NanoAODBaseCrossCleaner::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void NanoAODBaseCrossCleaner::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void NanoAODBaseCrossCleaner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NanoAODBaseCrossCleaner);
