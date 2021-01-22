#include "RecoEgamma/EgammaTools/interface/HGCalEgammaIDHelper.h"

#include <iostream>

HGCalEgammaIDHelper::HGCalEgammaIDHelper(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : eeRecHitInputTag_(iConfig.getParameter<edm::InputTag>("EERecHits")),
      fhRecHitInputTag_(iConfig.getParameter<edm::InputTag>("FHRecHits")),
      bhRecHitInputTag_(iConfig.getParameter<edm::InputTag>("BHRecHits")),
      hitMapInputTag_(iConfig.getParameter<edm::InputTag>("hitMapTag")),
      dEdXWeights_(iConfig.getParameter<std::vector<double>>("dEdXWeights")) {
  isoHelper_.setDeltaR(iConfig.getParameter<double>("isoDeltaR"));
  isoHelper_.setNRings(iConfig.getParameter<unsigned int>("isoNRings"));
  isoHelper_.setMinDeltaR(iConfig.getParameter<double>("isoDeltaRmin"));

  recHitsEE_ = iC.consumes<HGCRecHitCollection>(eeRecHitInputTag_);
  recHitsFH_ = iC.consumes<HGCRecHitCollection>(fhRecHitInputTag_);
  recHitsBH_ = iC.consumes<HGCRecHitCollection>(bhRecHitInputTag_);
  hitMap_ = iC.consumes<std::unordered_map<DetId, const HGCRecHit*>>(hitMapInputTag_);
  caloGeometry_ = iC.esConsumes();
  pcaHelper_.setdEdXWeights(dEdXWeights_);
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
}

void HGCalEgammaIDHelper::eventInit(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto recHitHandleEE = iEvent.getHandle(recHitsEE_);
  auto recHitHandleFH = iEvent.getHandle(recHitsFH_);
  auto recHitHandleBH = iEvent.getHandle(recHitsBH_);

  recHitTools_.setGeometry(iSetup.getData(caloGeometry_));
  pcaHelper_.setRecHitTools(&recHitTools_);
  isoHelper_.setRecHitTools(&recHitTools_);
  pcaHelper_.setHitMap(&iEvent.get(hitMap_));
  isoHelper_.setRecHits(recHitHandleEE, recHitHandleFH, recHitHandleBH);
}

void HGCalEgammaIDHelper::computeHGCAL(const reco::Photon& thePhoton, float radius) {
  if (thePhoton.isEB()) {
    if (debug_)
      std::cout << "The photon is in the barrel" << std::endl;
    pcaHelper_.clear();
    return;
  }

  pcaHelper_.storeRecHits(*thePhoton.superCluster()->seed());
  if (debug_)
    std::cout << " Stored the hits belonging to the photon superCluster seed " << std::endl;

  // initial computation, no radius cut, but halo hits not taken
  if (debug_)
    std::cout << " Calling PCA initial computation" << std::endl;
  pcaHelper_.pcaInitialComputation();
  // first computation within cylinder, halo hits included
  pcaHelper_.computePCA(radius);
  // second computation within cylinder, halo hits included
  pcaHelper_.computePCA(radius);
  pcaHelper_.computeShowerWidth(radius);

  // isolation
  isoHelper_.produceHGCalIso(thePhoton.superCluster()->seed());
}

void HGCalEgammaIDHelper::computeHGCAL(const reco::GsfElectron& theElectron, float radius) {
  if (theElectron.isEB()) {
    if (debug_)
      std::cout << "The electron is in the barrel" << std::endl;
    pcaHelper_.clear();
    return;
  }

  pcaHelper_.storeRecHits(*theElectron.electronCluster());
  if (debug_)
    std::cout << " Stored the hits belonging to the electronCluster " << std::endl;

  // initial computation, no radius cut, but halo hits not taken
  if (debug_)
    std::cout << " Calling PCA initial computation" << std::endl;
  pcaHelper_.pcaInitialComputation();
  // first computation within cylinder, halo hits included
  pcaHelper_.computePCA(radius);
  // second computation within cylinder, halo hits included
  pcaHelper_.computePCA(radius);
  pcaHelper_.computeShowerWidth(radius);
  isoHelper_.produceHGCalIso(theElectron.electronCluster());
}
