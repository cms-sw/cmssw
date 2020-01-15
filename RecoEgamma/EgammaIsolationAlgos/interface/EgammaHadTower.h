#ifndef EgammaHadTower_h
#define EgammaHadTower_h

// Finds the towers behind a super-cluster using the CaloTowerConstituentMap
// Florian Beaudette 22 Jun 2011

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class HcalChannelQuality;

class EgammaHadTower {
public:
  enum HoeMode { SingleTower = 0, TowersBehindCluster = 1 };

  EgammaHadTower(const edm::EventSetup& es, HoeMode mode = SingleTower);
  double getDepth1HcalESum(const std::vector<CaloTowerDetId>& towers, CaloTowerCollection const&) const;
  double getDepth2HcalESum(const std::vector<CaloTowerDetId>& towers, CaloTowerCollection const&) const;
  std::vector<CaloTowerDetId> towersOf(const reco::SuperCluster& sc) const;
  CaloTowerDetId towerOf(const reco::CaloCluster& cluster) const;
  bool hasActiveHcal(const std::vector<CaloTowerDetId>& towers) const;

private:
  const CaloTowerConstituentsMap* towerMap_;
  HoeMode mode_;
  unsigned int nMaxClusters_;
  const HcalChannelQuality* hcalQuality_;
  const HcalTopology* hcalTopology_;
};

#endif
