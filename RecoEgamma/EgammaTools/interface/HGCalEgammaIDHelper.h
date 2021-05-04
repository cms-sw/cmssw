//--------------------------------------------------------------------------------------------------
//
// EGammaID Helper
//
// Helper Class to compute HGCal Egamma cluster ID observables
//
// Authors: F. Beaudette, A. Lobanov, N. Smith
//--------------------------------------------------------------------------------------------------

#ifndef RecoEgamma_EgammaTools_HGCalEgammaIDHelper_h
#define RecoEgamma_EgammaTools_HGCalEgammaIDHelper_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "RecoEgamma/EgammaTools/interface/EgammaPCAHelper.h"
#include "RecoEgamma/EgammaTools/interface/LongDeps.h"
#include <vector>
#include "HGCalIsoCalculator.h"

class HGCalEgammaIDHelper {
public:
  HGCalEgammaIDHelper() {}
  HGCalEgammaIDHelper(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
  ~HGCalEgammaIDHelper() {}

  // Use eventInit once per event
  void eventInit(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  // Call computeHGCAL before accessing results below
  void computeHGCAL(const reco::Photon& thePhoton, float radius);
  void computeHGCAL(const reco::GsfElectron& theElectron, float radius);

  // PCA results
  double sigmaUU() const { return pcaHelper_.sigmaUU(); }
  double sigmaVV() const { return pcaHelper_.sigmaVV(); }
  double sigmaEE() const { return pcaHelper_.sigmaEE(); }
  double sigmaPP() const { return pcaHelper_.sigmaPP(); }
  const TVectorD& eigenValues() const { return pcaHelper_.eigenValues(); }
  const TVectorD& sigmas() const { return pcaHelper_.sigmas(); }
  const math::XYZPoint& barycenter() const { return pcaHelper_.barycenter(); }
  const math::XYZVector& axis() const { return pcaHelper_.axis(); }

  // longitudinal energy deposits and energy per subdetector as well as layers crossed
  hgcal::LongDeps energyPerLayer(float radius, bool withHalo = true) {
    return pcaHelper_.energyPerLayer(radius, withHalo);
  }

  // shower depth (distance between start and shower max) from ShowerDepth tool
  float clusterDepthCompatibility(const hgcal::LongDeps& ld, float& measDepth, float& expDepth, float& expSigma) {
    return pcaHelper_.clusterDepthCompatibility(ld, measDepth, expDepth, expSigma);
  }

  // projective calo isolation
  inline float getIsolationRing(unsigned int ring) const { return isoHelper_.getIso(ring); };

  // for debugging purposes
  void printHits(float radius) const { pcaHelper_.printHits(radius); }
  const hgcal::EGammaPCAHelper* pcaHelper() const { return &pcaHelper_; }

private:
  edm::InputTag eeRecHitInputTag_;
  edm::InputTag fhRecHitInputTag_;
  edm::InputTag bhRecHitInputTag_;
  edm::InputTag hitMapInputTag_;

  std::vector<double> dEdXWeights_;
  hgcal::EGammaPCAHelper pcaHelper_;
  HGCalIsoCalculator isoHelper_;
  edm::EDGetTokenT<HGCRecHitCollection> recHitsEE_;
  edm::EDGetTokenT<HGCRecHitCollection> recHitsFH_;
  edm::EDGetTokenT<HGCRecHitCollection> recHitsBH_;
  edm::EDGetTokenT<std::unordered_map<DetId, const HGCRecHit*>> hitMap_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_;
  hgcal::RecHitTools recHitTools_;
  bool debug_;
};

#endif
