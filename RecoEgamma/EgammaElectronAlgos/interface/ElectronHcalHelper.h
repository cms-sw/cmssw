#ifndef ElectronHcalHelper_h
#define ElectronHcalHelper_h

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

class ConsumesCollector;
class EgammaHadTower;
class HcalTopology;
class HcalChannelQuality;
class CaloTowerConstituentsMap;

class ElectronHcalHelper {
public:
  struct Configuration {
    // common parameters
    double hOverEConeSize;

    // strategy
    bool useTowers, checkHcalStatus;

    // specific parameters if use towers
    edm::EDGetTokenT<CaloTowerCollection> hcalTowers;
    double hOverEPtMin;  // min tower Et for H/E evaluation

    // specific parameters if use rechits
    edm::EDGetTokenT<HBHERecHitCollection> hcalRecHits;
    double hOverEHBMinE;
    double hOverEHFMinE;
  };

  ElectronHcalHelper(const Configuration &cfg, edm::ConsumesCollector &&cc);

  void beginEvent(const edm::Event &, const edm::EventSetup &);

  double hcalESum(const reco::SuperCluster &, const std::vector<CaloTowerDetId> *excludeTowers = nullptr) const;
  double hcalESumDepth1(const reco::SuperCluster &, const std::vector<CaloTowerDetId> *excludeTowers = nullptr) const;
  double hcalESumDepth2(const reco::SuperCluster &, const std::vector<CaloTowerDetId> *excludeTowers = nullptr) const;
  double hOverEConeSize() const { return cfg_.hOverEConeSize; }

  // Behind clusters
  inline auto hcalTowersBehindClusters(const reco::SuperCluster &sc) const { return egamma::towersOf(sc, *towerMap_); }
  inline auto hcalESumDepth1BehindClusters(const std::vector<CaloTowerDetId> &towers) const {
    return egamma::depth1HcalESum(towers, *towersFromCollection_);
  }
  inline auto hcalESumDepth2BehindClusters(const std::vector<CaloTowerDetId> &towers) const {
    return egamma::depth2HcalESum(towers, *towersFromCollection_);
  }

  // forward EgammaHadTower methods, if checkHcalStatus is enabled, using towers and H/E
  // otherwise, return true
  bool hasActiveHcal(const reco::SuperCluster &sc) const;

private:
  const Configuration cfg_;

  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> hcalChannelQualityToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalTopologyToken_;
  edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> towerMapToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  // event data (rechits strategy)
  std::unique_ptr<EgammaHcalIsolation> hcalIso_ = nullptr;

  // event data (towers strategy)
  std::unique_ptr<EgammaTowerIsolation> towerIso1_ = nullptr;
  std::unique_ptr<EgammaTowerIsolation> towerIso2_ = nullptr;
  CaloTowerCollection const *towersFromCollection_ = nullptr;
  CaloTowerConstituentsMap const *towerMap_ = nullptr;
  HcalChannelQuality const *hcalQuality_ = nullptr;
  HcalTopology const *hcalTopology_ = nullptr;
};

#endif
