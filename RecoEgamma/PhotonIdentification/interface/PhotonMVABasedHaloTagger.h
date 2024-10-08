/** \class PhotonMVABasedHaloTagger
 *   \author Shilpi Jain (University of Minnesota)
 * Links to the presentation: 
 1. ECAL DPG: https://indico.cern.ch/event/991261/contributions/4283096/attachments/2219229/3757719/beamHalo_31march_v1.pdf
 2. JetMET POG: https://indico.cern.ch/event/1027614/contributions/4314949/attachments/2224472/3767396/beamHalo_12April.pdf
 */

#ifndef RecoEgamma_PhotonIdentification_PhotonMVABasedHaloTagger_h
#define RecoEgamma_PhotonIdentification_PhotonMVABasedHaloTagger_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include <vector>

class PhotonMVABasedHaloTagger {
public:
  PhotonMVABasedHaloTagger(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);

  double calculateMVA(const reco::Photon* pho,
                      const GBRForest* gbr_,
                      const edm::Event& iEvent,
                      const edm::EventSetup& es) const;

private:
  struct CalClus {
    double x_ = 0.;
    double y_ = 0.;
    double z_ = 0.;
    double e_ = 0.;
    int nHits_ = 0;
  };

  struct HcalHyp {
    CalClus samedPhi_;
    CalClus samedR_;
  };

  struct PreshowerHyp {
    CalClus samedPhi_;
    CalClus samedR_;
  };

  struct EcalClus {
    double x_ = 0.;
    double y_ = 0.;
    double z_ = 0.;
    double e_ = 0.;
    int nHits_ = 0;
  };

  PreshowerHyp calmatchedESCoordForBothHypothesis(const CaloGeometry& geo,
                                                  const reco::Photon*,
                                                  const EcalRecHitCollection& ESRecHits,
                                                  const EcalClus& ecalClus) const;

  EcalClus calphoClusCoordinECAL(const CaloGeometry& geo,
                                 const reco::Photon*,
                                 const EcalPFRecHitThresholds* thresholds,
                                 const EcalRecHitCollection& ecalRecHits) const;
  HcalHyp calmatchedHBHECoordForBothHypothesis(const CaloGeometry& geo,
                                               const reco::Photon*,
                                               const HBHERecHitCollection& HBHERecHits,
                                               const EcalClus& ecalClus) const;
  double calAngleBetweenEEAndSubDet(CalClus const& subdet, EcalClus const&) const;

  double noiseThrES_;

  EgammaHcalIsolation::arrayHB recHitEThresholdHB_;
  EgammaHcalIsolation::arrayHE recHitEThresholdHE_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  const edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> ecalPFRechitThresholdsToken_;
  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  edm::EDGetTokenT<double> rhoLabel_;
  edm::EDGetTokenT<EcalRecHitCollection> EBecalCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> EEecalCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> ESCollection_;
  edm::EDGetTokenT<HBHERecHitCollection> HBHERecHitsCollection_;

  ///values of dR etc to cluster the hits in various sub-detectors
  static constexpr float dr2Max_ECALClus_ = 0.2 * 0.2;
  static constexpr float rho2Min_ECALpos_ = 31 * 31;            //cm
  static constexpr float rho2Max_ECALpos_ = 172 * 172;          //cm
  static constexpr float dRho2Max_HCALClus_SamePhi_ = 26 * 26;  //cm
  static constexpr float dPhiMax_HCALClus_SamePhi_ = 0.15;
  static constexpr float dR2Max_HCALClus_SamePhi_ = 0.15 * 0.15;
  static constexpr float dRho2Max_ESClus_ = 2.2 * 2.2;  //cm
  static constexpr float dXY_ESClus_SamePhi_ = 1;       ///cm
  static constexpr float dXY_ESClus_SamedR_ = 1;        ///cm
};

#endif  // PhotonMVABasedHaloTagger_H
