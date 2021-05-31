#ifndef ElectronHcalHelper_h
#define ElectronHcalHelper_h

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"

class ConsumesCollector;
class EgammaHadTower;
class HcalTopology;
class HcalChannelQuality;
class HcalHcalSeverityLevelComputer;
class CaloTowerConstituentsMap;

class ElectronHcalHelper {
public:
  struct Configuration {
    // common parameters
    double hOverEConeSize;

    // strategy
    bool onlyBehindCluster, checkHcalStatus;

    // specific parameters if use rechits
    edm::EDGetTokenT<HBHERecHitCollection> hbheRecHits;

    EgammaHcalIsolation::arrayHB eThresHB;
    int maxSeverityHB;
    EgammaHcalIsolation::arrayHE eThresHE;
    int maxSeverityHE;
  };

  ElectronHcalHelper(const Configuration &cfg, edm::ConsumesCollector &&cc);

  void beginEvent(const edm::Event &evt, const edm::EventSetup &eventSetup);

  inline auto hcalTowersBehindClusters(const reco::SuperCluster &sc) const { return egamma::towersOf(sc, *towerMap_); }
  double hcalESum(const reco::SuperCluster &, int depth) const;
  double hOverEConeSize() const { return cfg_.hOverEConeSize; }
  int maxSeverityHB() const { return cfg_.maxSeverityHB; }
  int maxSeverityHE() const { return cfg_.maxSeverityHE; }

  // forward EgammaHadTower methods, if checkHcalStatus is enabled, using towers and H/E
  bool hasActiveHcal(const reco::SuperCluster &sc) const;

  // QoL when one needs raw instances of EgammaHcalIsolation in addition to this class
  const auto hcalTopology() const { return hcalTopology_; }
  const auto hcalChannelQuality() const { return hcalChannelQuality_; }
  const auto hcalSevLvlComputer() const { return hcalSevLvlComputer_; }
  const auto towerMap() const { return towerMap_; }

private:
  const Configuration cfg_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalTopologyToken_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> hcalChannelQualityToken_;
  edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> hcalSevLvlComputerToken_;
  edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> towerMapToken_;

  // event data (rechits strategy)
  std::unique_ptr<EgammaHcalIsolation> hcalIso_;
  HcalTopology const *hcalTopology_ = nullptr;
  HcalChannelQuality const *hcalChannelQuality_ = nullptr;
  HcalSeverityLevelComputer const *hcalSevLvlComputer_ = nullptr;
  CaloTowerConstituentsMap const *towerMap_ = nullptr;
};

#endif
