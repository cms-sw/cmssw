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
  const auto& jetsProd = iEvent.get(jets_);
  std::vector<uint8_t> jets;
  jets.reserve(jetsProd.size());
  for (const auto& j : jetsProd) {
    jets.push_back(jetSel_(j));
  }
  auto jetsTable = std::make_unique<nanoaod::FlatTable>(jetsProd.size(), jetName_, false, true);

  const auto& muonsProd = iEvent.get(muons_);
  std::vector<uint8_t> muons;
  muons.reserve(muonsProd.size());
  for (const auto& m : muonsProd) {
    muons.push_back(muonSel_(m));
  }
  auto muonsTable = std::make_unique<nanoaod::FlatTable>(muonsProd.size(), muonName_, false, true);

  const auto& electronsProd = iEvent.get(electrons_);
  std::vector<uint8_t> eles;
  eles.reserve(electronsProd.size());
  for (const auto& e : electronsProd) {
    eles.push_back(electronSel_(e));
  }
  auto electronsTable = std::make_unique<nanoaod::FlatTable>(electronsProd.size(), electronName_, false, true);

  const auto& lowPtelectronsProd = iEvent.get(lowPtElectrons_);
  std::vector<uint8_t> lowPtEles;
  lowPtEles.reserve(lowPtelectronsProd.size());
  for (const auto& e : lowPtelectronsProd) {
    lowPtEles.push_back(lowPtElectronSel_(e));
  }
  auto lowPtElectronsTable = std::make_unique<nanoaod::FlatTable>(lowPtEles.size(), lowPtElectronName_, false, true);

  const auto& tausProd = iEvent.get(taus_);
  std::vector<uint8_t> taus;
  for (const auto& t : tausProd) {
    taus.push_back(tauSel_(t));
  }
  auto tausTable = std::make_unique<nanoaod::FlatTable>(tausProd.size(), tauName_, false, true);

  const auto& photonsProd = iEvent.get(photons_);
  std::vector<uint8_t> photons;
  for (const auto& p : photonsProd) {
    photons.push_back(photonSel_(p));
  }
  auto photonsTable = std::make_unique<nanoaod::FlatTable>(photonsProd.size(), photonName_, false, true);

  objectSelection(jetsProd, muonsProd, electronsProd, tausProd, photonsProd, jets, muons, eles, taus, photons);

  muonsTable->addColumn<uint8_t>(name_, muons, doc_);
  jetsTable->addColumn<uint8_t>(name_, jets, doc_);
  electronsTable->addColumn<uint8_t>(name_, eles, doc_);
  lowPtElectronsTable->addColumn<uint8_t>(name_, lowPtEles, doc_);
  tausTable->addColumn<uint8_t>(name_, taus, doc_);
  photonsTable->addColumn<uint8_t>(name_, photons, doc_);

  iEvent.put(std::move(jetsTable), "jets");
  iEvent.put(std::move(muonsTable), "muons");
  iEvent.put(std::move(electronsTable), "electrons");
  iEvent.put(std::move(tausTable), "taus");
  iEvent.put(std::move(photonsTable), "photons");
  iEvent.put(std::move(lowPtElectronsTable), "lowPtElectrons");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void NanoAODBaseCrossCleaner::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void NanoAODBaseCrossCleaner::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void NanoAODBaseCrossCleaner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name")->setComment("suffix name of the output flat table");
  desc.add<std::string>("doc")->setComment(
      "a bitmap defining the objects that remain after selection and cross cleaning");
  desc.add<edm::InputTag>("jets")->setComment("a jet collection derived from pat::Jet");
  desc.add<edm::InputTag>("muons")->setComment("a muon collection derived from pat::Muon");
  desc.add<edm::InputTag>("electrons")->setComment("an electron collection derived from pat::Electron");
  desc.add<edm::InputTag>("lowPtElectrons")
      ->setComment("an optional electron collection derived from pat::Electron, empty=>not used");
  desc.add<edm::InputTag>("taus")->setComment("a tau collection derived from pat::Tau");
  desc.add<edm::InputTag>("photons")->setComment("a photon collection derived from pat::Photon");

  desc.add<std::string>("jetSel")->setComment("function on pat::Jet defining the selection of jets");
  desc.add<std::string>("muonSel")->setComment("function on pat::Muon defining the selection of muons");
  desc.add<std::string>("electronSel")->setComment("function on pat::Electron defining the selection of electrons");
  desc.add<std::string>("lowPtElectronSel")
      ->setComment("function on pat::Electron defining the selection on alternative electrons collection");
  desc.add<std::string>("tauSel")->setComment("function on pat::Tau defining the selection on taus");
  desc.add<std::string>("photonSel")->setComment("function on pat::Photon defining the selection on photons");

  desc.add<std::string>("jetName")->setComment("name of the jet mask flat table output");
  desc.add<std::string>("muonName")->setComment("name of the muon mask flat table output");
  desc.add<std::string>("electronName")->setComment("name of the electron mask flat table output");
  desc.add<std::string>("lowPtElectronName")->setComment("name of the alternative electron mask flat table output");
  desc.add<std::string>("tauName")->setComment("name of the tau mask flat table output");
  desc.add<std::string>("photonName")->setComment("name of the photon mask flat table output");

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NanoAODBaseCrossCleaner);
