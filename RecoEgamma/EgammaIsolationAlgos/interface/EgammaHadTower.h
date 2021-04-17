#ifndef EgammaHadTower_h
#define EgammaHadTower_h

// Finds the towers behind a super-cluster using the CaloTowerConstituentMap
// Florian Beaudette 22 Jun 2011

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class HcalChannelQuality;

namespace egamma {
  enum class HoeMode { SingleTower = 0, TowersBehindCluster = 1 };

  double depth1HcalESum(std::vector<CaloTowerDetId> const& towers, CaloTowerCollection const&);
  double depth2HcalESum(std::vector<CaloTowerDetId> const& towers, CaloTowerCollection const&);

  std::vector<CaloTowerDetId> towersOf(reco::SuperCluster const& sc,
                                       CaloTowerConstituentsMap const& towerMap,
                                       HoeMode mode = HoeMode::SingleTower);

  CaloTowerDetId towerOf(reco::CaloCluster const& cluster, CaloTowerConstituentsMap const& towerMap);

  bool hasActiveHcal(std::vector<CaloTowerDetId> const& towers,
                     CaloTowerConstituentsMap const& towerMap,
                     HcalChannelQuality const& hcalQuality,
                     HcalTopology const& hcalTopology);
};  // namespace egamma

#endif
