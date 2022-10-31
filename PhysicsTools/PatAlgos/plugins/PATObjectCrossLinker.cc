// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      PATObjectCrossLinker
//
/**\class PATObjectCrossLinker PATObjectCrossLinker.cc PhysicsTools/PATObjectCrossLinker/plugins/PATObjectCrossLinker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Mon, 28 Aug 2017 09:26:39 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include "DataFormats/Common/interface/View.h"

#include "PhysicsTools/NanoAOD/interface/MatchingUtils.h"
//
// class declaration
//

class PATObjectCrossLinker : public edm::stream::EDProducer<> {
public:
  explicit PATObjectCrossLinker(const edm::ParameterSet&);
  ~PATObjectCrossLinker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  template <class C1, class C2, class C3, class C4>
  void matchOneToMany(const C1& refProdOne,
                      C2& itemsOne,
                      const std::string& nameOne,
                      const C3& refProdMany,
                      C4& itemsMany,
                      const std::string& nameMany);

  template <class C1, class C2, class C3, class C4>
  void matchElectronToPhoton(const C1& refProdOne,
                             C2& itemsOne,
                             const std::string& nameOne,
                             const C3& refProdMany,
                             C4& itemsMany,
                             const std::string& nameMany);

  template <class C1, class C2, class C3, class C4>
  void matchLowPtToElectron(const C1& refProdOne,
                            C2& itemsOne,
                            const std::string& nameOne,
                            const C3& refProdMany,
                            C4& itemsMany,
                            const std::string& nameMany);

  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<edm::View<pat::Jet>> jets_;
  const edm::EDGetTokenT<edm::View<pat::Muon>> muons_;
  const edm::EDGetTokenT<edm::View<pat::Electron>> electrons_;
  edm::InputTag lowPtElectronsTag_;
  edm::EDGetTokenT<edm::View<pat::Electron>> lowPtElectrons_;
  const edm::EDGetTokenT<edm::View<pat::Tau>> taus_;
  const edm::EDGetTokenT<edm::View<pat::Photon>> photons_;
};

//
// constructors and destructor
//
PATObjectCrossLinker::PATObjectCrossLinker(const edm::ParameterSet& params)
    : jets_(consumes<edm::View<pat::Jet>>(params.getParameter<edm::InputTag>("jets"))),
      muons_(consumes<edm::View<pat::Muon>>(params.getParameter<edm::InputTag>("muons"))),
      electrons_(consumes<edm::View<pat::Electron>>(params.getParameter<edm::InputTag>("electrons"))),
      lowPtElectronsTag_(params.getParameter<edm::InputTag>("lowPtElectrons")),
      lowPtElectrons_(mayConsume<edm::View<pat::Electron>>(lowPtElectronsTag_)),
      taus_(consumes<edm::View<pat::Tau>>(params.getParameter<edm::InputTag>("taus"))),
      photons_(consumes<edm::View<pat::Photon>>(params.getParameter<edm::InputTag>("photons")))

{
  produces<std::vector<pat::Jet>>("jets");
  produces<std::vector<pat::Muon>>("muons");
  produces<std::vector<pat::Electron>>("electrons");
  if (!lowPtElectronsTag_.label().empty())
    produces<std::vector<pat::Electron>>("lowPtElectrons");
  produces<std::vector<pat::Tau>>("taus");
  produces<std::vector<pat::Photon>>("photons");
}

PATObjectCrossLinker::~PATObjectCrossLinker() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------

///
template <class C1, class C2, class C3, class C4>
void PATObjectCrossLinker::matchOneToMany(const C1& refProdOne,
                                          C2& itemsOne,
                                          const std::string& nameOne,
                                          const C3& refProdMany,
                                          C4& itemsMany,
                                          const std::string& nameMany) {
  size_t ji = 0;
  for (auto& j : itemsOne) {
    edm::PtrVector<reco::Candidate> overlaps(refProdMany.id());
    size_t mi = 0;
    for (auto& m : itemsMany) {
      if (matchByCommonSourceCandidatePtr(j, m) && (!m.hasUserCand(nameOne))) {
        m.addUserCand(nameOne, reco::CandidatePtr(refProdOne.id(), ji, refProdOne.productGetter()));
        overlaps.push_back(reco::CandidatePtr(refProdMany.id(), mi, refProdMany.productGetter()));
      }
      mi++;
    }
    j.setOverlaps(nameMany, overlaps);
    ji++;
  }
}

template <class C1, class C2, class C3, class C4>
void PATObjectCrossLinker::matchElectronToPhoton(const C1& refProdOne,
                                                 C2& itemsOne,
                                                 const std::string& nameOne,
                                                 const C3& refProdMany,
                                                 C4& itemsMany,
                                                 const std::string& nameMany) {
  size_t ji = 0;
  for (auto& j : itemsOne) {
    edm::PtrVector<reco::Candidate> overlaps(refProdMany.id());
    size_t mi = 0;
    for (auto& m : itemsMany) {
      if (matchByCommonParentSuperClusterRef(j, m) && (!m.hasUserCand(nameOne))) {
        m.addUserCand(nameOne, reco::CandidatePtr(refProdOne.id(), ji, refProdOne.productGetter()));
        overlaps.push_back(reco::CandidatePtr(refProdMany.id(), mi, refProdMany.productGetter()));
      }
      mi++;
    }
    j.setOverlaps(nameMany, overlaps);
    ji++;
  }
}

template <class C1, class C2, class C3, class C4>
void PATObjectCrossLinker::matchLowPtToElectron(const C1& refProdOne,
                                                C2& itemsOne,
                                                const std::string& nameOne,
                                                const C3& refProdMany,
                                                C4& itemsMany,
                                                const std::string& nameMany) {
  size_t ji = 0;
  for (auto& j : itemsOne) {
    std::vector<std::pair<size_t, float>> idxs;
    size_t mi = 0;
    for (auto& m : itemsMany) {
      float dr2 = deltaR2(m, j);
      if (dr2 < 1.e-6) {  // deltaR < 1.e-3
        m.addUserCand(nameOne, reco::CandidatePtr(refProdOne.id(), ji, refProdOne.productGetter()));
        idxs.push_back(std::make_pair(mi, dr2));
      }
      mi++;
    }
    std::sort(idxs.begin(), idxs.end(), [](auto& left, auto& right) { return left.second < right.second; });

    edm::PtrVector<reco::Candidate> overlaps(refProdMany.id());
    for (auto idx : idxs) {
      overlaps.push_back(reco::CandidatePtr(refProdMany.id(), idx.first, refProdMany.productGetter()));
    }
    j.setOverlaps(nameMany, overlaps);
    ji++;
  }
}

void PATObjectCrossLinker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  const auto& jetsIn = iEvent.get(jets_);
  auto jets = std::make_unique<std::vector<pat::Jet>>();
  jets->reserve(jetsIn.size());
  for (const auto& j : jetsIn)
    jets->push_back(j);
  auto jetRefProd = iEvent.getRefBeforePut<std::vector<pat::Jet>>("jets");

  const auto& muonsIn = iEvent.get(muons_);
  auto muons = std::make_unique<std::vector<pat::Muon>>();
  muons->reserve(muonsIn.size());
  for (const auto& m : muonsIn)
    muons->push_back(m);
  auto muRefProd = iEvent.getRefBeforePut<std::vector<pat::Muon>>("muons");

  const auto& electronsIn = iEvent.get(electrons_);
  auto electrons = std::make_unique<std::vector<pat::Electron>>();
  electrons->reserve(electronsIn.size());
  for (const auto& e : electronsIn)
    electrons->push_back(e);
  auto eleRefProd = iEvent.getRefBeforePut<std::vector<pat::Electron>>("electrons");

  auto lowPtElectrons = std::make_unique<std::vector<pat::Electron>>();
  if (!lowPtElectronsTag_.label().empty()) {
    auto lowPtElectronsIn = iEvent.getHandle(lowPtElectrons_);
    lowPtElectrons->reserve(lowPtElectronsIn->size());
    for (const auto& e : *lowPtElectronsIn) {
      lowPtElectrons->push_back(e);
    }
  }

  const auto& tausIn = iEvent.get(taus_);
  auto taus = std::make_unique<std::vector<pat::Tau>>();
  taus->reserve(tausIn.size());
  for (const auto& t : tausIn)
    taus->push_back(t);
  auto tauRefProd = iEvent.getRefBeforePut<std::vector<pat::Tau>>("taus");

  const auto& photonsIn = iEvent.get(photons_);
  auto photons = std::make_unique<std::vector<pat::Photon>>();
  photons->reserve(photonsIn.size());
  for (const auto& p : photonsIn)
    photons->push_back(p);
  auto phRefProd = iEvent.getRefBeforePut<std::vector<pat::Photon>>("photons");

  matchOneToMany(jetRefProd, *jets, "jet", muRefProd, *muons, "muons");
  matchOneToMany(jetRefProd, *jets, "jet", eleRefProd, *electrons, "electrons");
  matchOneToMany(jetRefProd, *jets, "jet", tauRefProd, *taus, "taus");
  matchOneToMany(jetRefProd, *jets, "jet", phRefProd, *photons, "photons");

  matchElectronToPhoton(eleRefProd, *electrons, "electron", phRefProd, *photons, "photons");
  if (!lowPtElectronsTag_.label().empty()) {
    auto lowPtEleRefProd = iEvent.getRefBeforePut<std::vector<pat::Electron>>("lowPtElectrons");
    matchLowPtToElectron(lowPtEleRefProd, *lowPtElectrons, "lowPtElectron", eleRefProd, *electrons, "electrons");
  }

  iEvent.put(std::move(jets), "jets");
  iEvent.put(std::move(muons), "muons");
  iEvent.put(std::move(electrons), "electrons");
  if (!lowPtElectronsTag_.label().empty())
    iEvent.put(std::move(lowPtElectrons), "lowPtElectrons");
  iEvent.put(std::move(taus), "taus");
  iEvent.put(std::move(photons), "photons");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void PATObjectCrossLinker::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void PATObjectCrossLinker::endStream() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PATObjectCrossLinker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets")->setComment("a jet collection derived from pat::Jet");
  desc.add<edm::InputTag>("muons")->setComment("a muon collection derived from pat::Muon");
  desc.add<edm::InputTag>("electrons")->setComment("an electron collection derived from pat::Electron");
  desc.add<edm::InputTag>("lowPtElectrons")
      ->setComment("an optional electron collection derived from pat::Electron, empty=>not used");
  desc.add<edm::InputTag>("taus")->setComment("a tau collection derived from pat::Tau");
  desc.add<edm::InputTag>("photons")->setComment("a photon collection derived from pat::Photon");
  descriptions.add("patObjectCrossLinker", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PATObjectCrossLinker);
