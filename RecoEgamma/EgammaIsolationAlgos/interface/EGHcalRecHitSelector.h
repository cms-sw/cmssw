#ifndef RecoEgamam_EgammaIsolationAlgos_EGHcalRecHitSelector_h
#define RecoEgamam_EgammaIsolationAlgos_EGHcalRecHitSelector_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

class EGHcalRecHitSelector {
public:
  explicit EGHcalRecHitSelector(const edm::ParameterSet& config, edm::ConsumesCollector);
  ~EGHcalRecHitSelector() {}

  template <typename CollType>
  void addDetIds(const reco::SuperCluster& superClus,
                 const HBHERecHitCollection& recHits,
                 CollType& detIdsToStore) const;

  void setup(const edm::EventSetup& iSetup) { towerMap_ = iSetup.getHandle(towerMapToken_); }

  static edm::ParameterSetDescription makePSetDescription();

private:
  static int calDIPhi(int iPhi1, int iPhi2);
  static int calDIEta(int iEta1, int iEta2);
  float getMinEnergyHCAL_(HcalDetId id) const;

  int maxDIEta_;
  int maxDIPhi_;
  float minEnergyHB_;
  float minEnergyHEDepth1_;
  float minEnergyHEDefault_;

  edm::ESHandle<CaloTowerConstituentsMap> towerMap_;
  edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> towerMapToken_;
};

template <typename CollType>
void EGHcalRecHitSelector::addDetIds(const reco::SuperCluster& superClus,
                                     const HBHERecHitCollection& recHits,
                                     CollType& detIdsToStore) const {
  DetId seedId = superClus.seed()->seed();
  if (seedId.det() != DetId::Ecal && seedId.det() != DetId::Forward) {
    edm::LogError("EgammaIsoHcalDetIdCollectionProducerError")
        << "Somehow the supercluster has a seed which is not ECAL, something is badly wrong";
    return;
  }
  //so we are using CaloTowers to get the iEta/iPhi of the HCAL rec hit behind the seed cluster, this might go funny on tower 28 but shouldnt matter there

  if (seedId.det() == DetId::Forward)
    return;

  CaloTowerDetId towerId(towerMap_->towerOf(seedId));
  int seedHcalIEta = towerId.ieta();
  int seedHcalIPhi = towerId.iphi();

  for (auto& recHit : recHits) {
    int dIEtaAbs = std::abs(calDIEta(seedHcalIEta, recHit.id().ieta()));
    int dIPhiAbs = std::abs(calDIPhi(seedHcalIPhi, recHit.id().iphi()));

    if (dIEtaAbs <= maxDIEta_ && dIPhiAbs <= maxDIPhi_ && recHit.energy() > getMinEnergyHCAL_(recHit.id()))
      detIdsToStore.insert(detIdsToStore.end(), recHit.id().rawId());
  }
}

#endif
