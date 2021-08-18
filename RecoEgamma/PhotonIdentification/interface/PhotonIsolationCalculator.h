#ifndef PhotonIsolationCalculator_H
#define PhotonIsolationCalculator_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"

class PhotonIsolationCalculator {
public:
  PhotonIsolationCalculator() {}

  ~PhotonIsolationCalculator() {}

  void setup(const edm::ParameterSet& conf,
             std::vector<int> const& flagsEB_,
             std::vector<int> const& flagsEE_,
             std::vector<int> const& severitiesEB_,
             std::vector<int> const& severitiesEE_,
             edm::ConsumesCollector&& iC);

  void calculate(const reco::Photon*,
                 const edm::Event&,
                 const edm::EventSetup& es,
                 reco::Photon::FiducialFlags& phofid,
                 reco::Photon::IsolationVariables& phoisolR03,
                 reco::Photon::IsolationVariables& phoisolR04) const;

private:
  static void classify(const reco::Photon* photon,
                       bool& isEBPho,
                       bool& isEEPho,
                       bool& isEBEtaGap,
                       bool& isEBPhiGap,
                       bool& isEERingGap,
                       bool& isEEDeeGap,
                       bool& isEBEEGap) dso_internal;

  void calculateTrackIso(const reco::Photon* photon,
                         const edm::Event& e,
                         double& trkCone,
                         int& ntrkCone,
                         double pTThresh = 0,
                         double RCone = .4,
                         double RinnerCone = .1,
                         double etaSlice = 0.015,
                         double lip = 0.2,
                         double d0 = 0.1) const dso_internal;

  double calculateEcalRecHitIso(const reco::Photon* photon,
                                const edm::Event& iEvent,
                                const edm::EventSetup& iSetup,
                                double RCone,
                                double RConeInner,
                                double etaSlice,
                                double eMin,
                                double etMin,
                                bool vetoClusteredHits,
                                bool useNumCrystals) const dso_internal;

  template <bool isoBC>
  double calculateHcalRecHitIso(const reco::Photon* photon,
                                const CaloGeometry& geometry,
                                const HcalTopology& hcalTopology,
                                const HcalChannelQuality& hcalChStatus,
                                const HcalSeverityLevelComputer& hcalSevLvlComputer,
                                const CaloTowerConstituentsMap& towerMap,
                                const HBHERecHitCollection& hbheRecHits,
                                double RCone,
                                double RConeInner,
                                int depth) const dso_internal;

private:
  edm::EDGetToken barrelecalCollection_;
  edm::EDGetToken endcapecalCollection_;
  edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitsTag_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalTopologyToken_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> hcalChannelQualityToken_;
  edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> hcalSevLvlComputerToken_;
  edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> towerMapToken_;

  edm::EDGetToken trackInputTag_;
  edm::EDGetToken beamSpotProducerTag_;
  double modulePhiBoundary_;
  std::vector<double> moduleEtaBoundary_;
  bool vetoClusteredEcalHits_;
  bool useNumCrystals_;

  std::array<double, 6> trkIsoBarrelRadiusA_;
  std::array<double, 5> ecalIsoBarrelRadiusA_;
  std::array<double, 6> trkIsoBarrelRadiusB_;
  std::array<double, 5> ecalIsoBarrelRadiusB_;

  std::array<double, 6> trkIsoEndcapRadiusA_;
  std::array<double, 5> ecalIsoEndcapRadiusA_;
  std::array<double, 6> trkIsoEndcapRadiusB_;
  std::array<double, 5> ecalIsoEndcapRadiusB_;

  std::array<double, 7> hcalIsoInnerRadAEB_;
  std::array<double, 7> hcalIsoOuterRadAEB_;

  std::array<double, 7> hcalIsoInnerRadBEB_;
  std::array<double, 7> hcalIsoOuterRadBEB_;

  std::array<double, 7> hcalIsoInnerRadAEE_;
  std::array<double, 7> hcalIsoOuterRadAEE_;

  std::array<double, 7> hcalIsoInnerRadBEE_;
  std::array<double, 7> hcalIsoOuterRadBEE_;

  EgammaHcalIsolation::arrayHB hcalIsoEThresHB_;
  EgammaHcalIsolation::arrayHE hcalIsoEThresHE_;
  int maxHcalSeverity_;

  std::vector<int> flagsEB_;
  std::vector<int> flagsEE_;
  std::vector<int> severityExclEB_;
  std::vector<int> severityExclEE_;
};

#endif  // PhotonIsolationCalculator_H
