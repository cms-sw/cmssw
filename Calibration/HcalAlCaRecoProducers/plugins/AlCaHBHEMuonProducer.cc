// -*- C++ -*-
//#define EDM_ML_DEBUG

// system include files
#include <atomic>
#include <memory>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <boost/regex.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

//#define EDM_ML_DEBUG
//
// class declaration
//

namespace alCaHBHEMuonProducer {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}  // namespace alCaHBHEMuonProducer

class AlCaHBHEMuonProducer : public edm::stream::EDProducer<edm::GlobalCache<alCaHBHEMuonProducer::Counters> > {
public:
  explicit AlCaHBHEMuonProducer(edm::ParameterSet const&, const alCaHBHEMuonProducer::Counters* count);
  ~AlCaHBHEMuonProducer() override;

  static std::unique_ptr<alCaHBHEMuonProducer::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<alCaHBHEMuonProducer::Counters>();
  }

  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  static void globalEndJob(const alCaHBHEMuonProducer::Counters* counters);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  bool select(const reco::MuonCollection&);

  // ----------member data ---------------------------
  unsigned int nRun_, nAll_, nGood_;
  const edm::InputTag labelBS_, labelVtx_;
  const edm::InputTag labelEB_, labelEE_, labelHBHE_, labelMuon_;
  const double pMuonMin_;

  edm::EDGetTokenT<reco::BeamSpot> tok_BS_;
  edm::EDGetTokenT<reco::VertexCollection> tok_Vtx_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_HBHE_;
  edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;
};

AlCaHBHEMuonProducer::AlCaHBHEMuonProducer(edm::ParameterSet const& iConfig,
                                           const alCaHBHEMuonProducer::Counters* count)
    : nRun_(0),
      nAll_(0),
      nGood_(0),
      labelBS_(iConfig.getParameter<edm::InputTag>("BeamSpotLabel")),
      labelVtx_(iConfig.getParameter<edm::InputTag>("VertexLabel")),
      labelEB_(iConfig.getParameter<edm::InputTag>("EBRecHitLabel")),
      labelEE_(iConfig.getParameter<edm::InputTag>("EERecHitLabel")),
      labelHBHE_(iConfig.getParameter<edm::InputTag>("HBHERecHitLabel")),
      labelMuon_(iConfig.getParameter<edm::InputTag>("MuonLabel")),
      pMuonMin_(iConfig.getParameter<double>("MinimumMuonP")) {
  // define tokens for access
  tok_Vtx_ = consumes<reco::VertexCollection>(labelVtx_);
  tok_BS_ = consumes<reco::BeamSpot>(labelBS_);
  tok_EB_ = consumes<EcalRecHitCollection>(labelEB_);
  tok_EE_ = consumes<EcalRecHitCollection>(labelEE_);
  tok_HBHE_ = consumes<HBHERecHitCollection>(labelHBHE_);
  tok_Muon_ = consumes<reco::MuonCollection>(labelMuon_);

  edm::LogVerbatim("HcalHBHEMuon") << "Parameters read from config file \n"
                                   << "\t minP of muon " << pMuonMin_ << "\t input labels " << labelBS_ << " "
                                   << labelVtx_ << " " << labelEB_ << " " << labelEE_ << " " << labelHBHE_ << " "
                                   << labelMuon_;

  //saves the following collections
  produces<reco::BeamSpot>(labelBS_.label());
  produces<reco::VertexCollection>(labelVtx_.label());
  produces<EcalRecHitCollection>(labelEB_.instance());
  produces<EcalRecHitCollection>(labelEE_.instance());
  produces<HBHERecHitCollection>(labelHBHE_.label());
  produces<reco::MuonCollection>(labelMuon_.label());
}

AlCaHBHEMuonProducer::~AlCaHBHEMuonProducer() {}

void AlCaHBHEMuonProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  ++nAll_;
  bool valid(true);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalHBHEMuon") << "AlCaHBHEMuonProducer::Run " << iEvent.id().run() << " Event "
                                   << iEvent.id().event() << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                   << iEvent.bunchCrossing();
#endif

  //Step1: Get all the relevant containers
  auto bmspot = iEvent.getHandle(tok_BS_);
  if (!bmspot.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelBS_;
    valid = false;
  }

  auto vt = iEvent.getHandle(tok_Vtx_);
  if (!vt.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelVtx_;
    valid = false;
  }

  auto barrelRecHitsHandle = iEvent.getHandle(tok_EB_);
  if (!barrelRecHitsHandle.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelEB_;
    valid = false;
  }

  auto endcapRecHitsHandle = iEvent.getHandle(tok_EE_);
  if (!endcapRecHitsHandle.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelEE_;
    valid = false;
  }

  auto hbhe = iEvent.getHandle(tok_HBHE_);
  if (!hbhe.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelHBHE_;
    valid = false;
  }

  auto muonhandle = iEvent.getHandle(tok_Muon_);
  if (!muonhandle.isValid()) {
    edm::LogWarning("HcalHBHEMuon") << "AlCaHBHEMuonProducer: Error! can't get product " << labelMuon_;
    valid = false;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalHBHEMuon") << "AlCaHBHEMuonProducer::obtained the collections with validity flag " << valid;
#endif

  //For accepted events
  auto outputBeamSpot = std::make_unique<reco::BeamSpot>();
  auto outputVColl = std::make_unique<reco::VertexCollection>();
  auto outputEBColl = std::make_unique<EBRecHitCollection>();
  auto outputEEColl = std::make_unique<EERecHitCollection>();
  auto outputHBHEColl = std::make_unique<HBHERecHitCollection>();
  auto outputMColl = std::make_unique<reco::MuonCollection>();

  if (valid) {
    const reco::BeamSpot beam = *(bmspot.product());
    outputBeamSpot = std::make_unique<reco::BeamSpot>(
        beam.position(), beam.sigmaZ(), beam.dxdz(), beam.dydz(), beam.BeamWidthX(), beam.covariance(), beam.type());
    const reco::VertexCollection vtx = *(vt.product());
    const EcalRecHitCollection ebcoll = *(barrelRecHitsHandle.product());
    const EcalRecHitCollection eecoll = *(endcapRecHitsHandle.product());
    const HBHERecHitCollection hbhecoll = *(hbhe.product());
    const reco::MuonCollection muons = *(muonhandle.product());

    bool accept = select(muons);

    if (accept) {
      ++nGood_;

      for (reco::VertexCollection::const_iterator vtr = vtx.begin(); vtr != vtx.end(); ++vtr)
        outputVColl->push_back(*vtr);

      for (edm::SortedCollection<EcalRecHit>::const_iterator ehit = ebcoll.begin(); ehit != ebcoll.end(); ++ehit)
        outputEBColl->push_back(*ehit);

      for (edm::SortedCollection<EcalRecHit>::const_iterator ehit = eecoll.begin(); ehit != eecoll.end(); ++ehit)
        outputEEColl->push_back(*ehit);

      for (std::vector<HBHERecHit>::const_iterator hhit = hbhecoll.begin(); hhit != hbhecoll.end(); ++hhit)
        outputHBHEColl->push_back(*hhit);

      for (reco::MuonCollection::const_iterator muon = muons.begin(); muon != muons.end(); ++muon)
        outputMColl->push_back(*muon);
    }
  }

  iEvent.put(std::move(outputBeamSpot), labelBS_.label());
  iEvent.put(std::move(outputVColl), labelVtx_.label());
  iEvent.put(std::move(outputEBColl), labelEB_.instance());
  iEvent.put(std::move(outputEEColl), labelEE_.instance());
  iEvent.put(std::move(outputHBHEColl), labelHBHE_.label());
  iEvent.put(std::move(outputMColl), labelMuon_.label());
}

void AlCaHBHEMuonProducer::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaHBHEMuonProducer::globalEndJob(const alCaHBHEMuonProducer::Counters* count) {
  edm::LogVerbatim("HcalHBHEMuon") << "Finds " << count->nGood_ << " good tracks in " << count->nAll_ << " events";
}

void AlCaHBHEMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("BeamSpotLabel", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("VertexLabel", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("EBRecHitLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("EERecHitLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("HBHERecHitLabel", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("MuonLabel", edm::InputTag("muons"));
  desc.add<double>("MinimumMuonP", 5.0);
  descriptions.add("alcaHBHEMuonProducer", desc);
}

void AlCaHBHEMuonProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogVerbatim("HcalHBHEMuon") << "Run[" << nRun_ << "] " << iRun.run();
}

void AlCaHBHEMuonProducer::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogVerbatim("HcalHBHEMuon") << "endRun[" << nRun_ << "] " << iRun.run();
}

bool AlCaHBHEMuonProducer::select(const reco::MuonCollection& muons) {
  bool ok(false);
  for (unsigned int k = 0; k < muons.size(); ++k) {
    if (muons[k].p() > pMuonMin_) {
      ok = true;
      break;
    }
  }
  return ok;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AlCaHBHEMuonProducer);
