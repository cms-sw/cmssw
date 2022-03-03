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
                      const edm::EventSetup& es);

private:
  void calphoClusCoordinECAL(const CaloGeometry* geo,
                             const reco::Photon*,
                             const EcalPFRecHitThresholds* thresholds,
                             const EcalRecHitCollection& ecalRecHits);

  void calmatchedHBHECoordForBothHypothesis(const CaloGeometry* geo,
                                            const reco::Photon*,
                                            const HBHERecHitCollection& HBHERecHits);

  void calmatchedESCoordForBothHypothesis(const CaloGeometry* geo,
                                          const reco::Photon*,
                                          const EcalRecHitCollection& ESRecHits);

  double calAngleBetweenEEAndSubDet(int nhits, double subdetClusX, double subdetClusY, double subdetClusZ);

  int hcalClusNhits_samedPhi_, hcalClusNhits_samedR_;
  int ecalClusNhits_, preshowerNhits_samedPhi_, preshowerNhits_samedR_;
  double hcalClusX_samedPhi_, hcalClusY_samedPhi_, hcalClusZ_samedPhi_, hcalClusX_samedR_, hcalClusY_samedR_,
      hcalClusZ_samedR_;
  double hcalClusE_samedPhi_, hcalClusE_samedR_;

  double ecalClusX_, ecalClusY_, ecalClusZ_;
  double preshowerX_samedPhi_, preshowerY_samedPhi_, preshowerZ_samedPhi_, preshowerX_samedR_, preshowerY_samedR_,
      preshowerZ_samedR_;
  double ecalClusE_, preshowerE_samedPhi_, preshowerE_samedR_;
  double noiseThrES_;

  EgammaHcalIsolation::arrayHB recHitEThresholdHB_;
  EgammaHcalIsolation::arrayHE recHitEThresholdHE_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  const edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> ecalPFRechitThresholdsToken_;
  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  edm::ESHandle<CaloGeometry> pG_;
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
