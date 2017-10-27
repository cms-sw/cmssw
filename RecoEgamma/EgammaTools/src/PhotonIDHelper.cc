#include "RecoEgamma/EgammaTools/interface/PhotonIDHelper.h"

#include <iostream>

PhotonIDHelper::PhotonIDHelper(const edm::ParameterSet  & iConfig,edm::ConsumesCollector && iC):
        eeRecHitInputTag_(iConfig.getParameter<edm::InputTag> ("EERecHits") ),
        fhRecHitInputTag_(iConfig.getParameter<edm::InputTag> ("FHRecHits") ),
        bhRecHitInputTag_(iConfig.getParameter<edm::InputTag> ("BHRecHits") ),
        dEdXWeights_(iConfig.getParameter<std::vector<double> >("dEdXWeights"))
{
    isoHelper_.setDeltaR(iConfig.getUntrackedParameter<double>("photonIsoDeltaR", 0.15));
    isoHelper_.setNRings(iConfig.getUntrackedParameter<int>("photonIsoNRings", 5));
    isoHelper_.setMinDeltaR(iConfig.getUntrackedParameter<double>("photonIsoDeltaRmin", 0.));

    recHitsEE_ = iC.consumes<HGCRecHitCollection>(eeRecHitInputTag_);
    recHitsFH_ = iC.consumes<HGCRecHitCollection>(fhRecHitInputTag_);
    recHitsBH_ = iC.consumes<HGCRecHitCollection>(bhRecHitInputTag_);
    pcaHelper_.setdEdXWeights(dEdXWeights_);
    debug_ = false;
}

void PhotonIDHelper::eventInit(const edm::Event& iEvent,const edm::EventSetup &iSetup) {
    edm::Handle<HGCRecHitCollection> recHitHandleEE;
    iEvent.getByToken(recHitsEE_, recHitHandleEE);
    edm::Handle<HGCRecHitCollection> recHitHandleFH;
    iEvent.getByToken(recHitsFH_, recHitHandleFH);
    edm::Handle<HGCRecHitCollection> recHitHandleBH;
    iEvent.getByToken(recHitsBH_, recHitHandleBH);

    recHitTools_.getEventSetup(iSetup);
    pcaHelper_.setRecHitTools(&recHitTools_);
    isoHelper_.setRecHitTools(&recHitTools_);
    pcaHelper_.fillHitMap(*recHitHandleEE,*recHitHandleFH,*recHitHandleBH);
    isoHelper_.setRecHits(recHitHandleEE, recHitHandleFH, recHitHandleBH);
}

void PhotonIDHelper::computeHGCAL(const reco::Photon & thePhoton, float radius) {
    thePhoton_ = &thePhoton;
    if (thePhoton.isEB()) {
        if (debug_) std::cout << "The photon is in the barrel" <<std::endl;
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
