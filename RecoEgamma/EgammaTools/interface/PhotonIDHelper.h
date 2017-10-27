//--------------------------------------------------------------------------------------------------
//
// EGammaID Helper
//
// Helper Class to compute photon ID observables
//
// Authors: F. Beaudette,A. Lobanov
//--------------------------------------------------------------------------------------------------

#ifndef PhotonIDHelper_H
#define PhotonIDHelper_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "RecoEgamma/EgammaTools/interface/EgammaPCAHelper.h"
#include "RecoEgamma/EgammaTools/interface/LongDeps.h"
#include <vector>
#include "HGCalIsoProducer.h"

class PhotonIDHelper {
public:
    PhotonIDHelper(){;}
    PhotonIDHelper(const edm::ParameterSet &, edm::ConsumesCollector && iC);
    ~PhotonIDHelper(){;}

    // Use eventInit once per event
    void eventInit(const edm::Event& iEvent,const edm::EventSetup &iSetup);

    void computeHGCAL(const reco::Photon & thePhoton, float radius);

    inline double photonClusterEnergy() const { return thePhoton_->superCluster()->seed()->energy();}

    inline double photonSCEnergy() const {return thePhoton_->superCluster()->energy();}

    inline double sigmaUU() const {  return pcaHelper_.sigmaUU();}
    inline double sigmaVV() const {  return pcaHelper_.sigmaVV();}
    inline double sigmaEE() const {  return pcaHelper_.sigmaEE();}
    inline double sigmaPP() const {  return pcaHelper_.sigmaPP();}
    inline TVectorD eigenValues () const {return pcaHelper_.eigenValues();}
    inline TVectorD sigmas() const {return pcaHelper_.sigmas();}


    // longitudinal energy deposits and energy per subdetector as well as layers crossed
    LongDeps energyPerLayer(float radius, bool withHalo=true) {
        return pcaHelper_.energyPerLayer(radius,withHalo);
    }

    const math::XYZPoint  & barycenter() const {return pcaHelper_.barycenter();}
    const math::XYZVector & axis() const {return pcaHelper_.axis();}
    void printHits(float radius) const { pcaHelper_.printHits(radius); }

    float clusterDepthCompatibility(const LongDeps & ld, float & measDepth, float & expDepth, float & expSigma)
        { return pcaHelper_.clusterDepthCompatibility(ld,measDepth,expDepth,expSigma);}

    inline float getIsolationRing(size_t ring) const { return isoHelper_.getIso(ring); };

    /// for debugging purposes, if you have to use it, it means that an interface method is missing
    EGammaPCAHelper * pcaHelper () {return &pcaHelper_;}

private:
    const reco::Photon * thePhoton_;
    edm::InputTag  eeRecHitInputTag_;
    edm::InputTag  fhRecHitInputTag_;
    edm::InputTag  bhRecHitInputTag_;

    std::vector<double> dEdXWeights_;
    EGammaPCAHelper pcaHelper_;
    HGCalIsoProducer isoHelper_;
    edm::EDGetTokenT<HGCRecHitCollection> recHitsEE_;
    edm::EDGetTokenT<HGCRecHitCollection> recHitsFH_;
    edm::EDGetTokenT<HGCRecHitCollection> recHitsBH_;
    hgcal::RecHitTools recHitTools_;
    bool debug_;
};

#endif
