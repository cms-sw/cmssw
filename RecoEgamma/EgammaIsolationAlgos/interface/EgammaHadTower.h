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

class HcalChannelQuality;

class EgammaHadTower {
 public:

  enum HoeMode{SingleTower=0,TowersBehindCluster=1};

  EgammaHadTower(const edm::EventSetup &es,HoeMode mode=SingleTower);
  ~EgammaHadTower(){;}
  double getDepth1HcalESum( const reco::SuperCluster & sc ) const ;
  double getDepth2HcalESum( const reco::SuperCluster & sc ) const ;
  double getDepth1HcalESum( const std::vector<CaloTowerDetId> & towers ) const ;
  double getDepth2HcalESum( const std::vector<CaloTowerDetId> & towers ) const ;
  std::vector<CaloTowerDetId> towersOf(const reco::SuperCluster& sc) const ;
  CaloTowerDetId  towerOf(const reco::CaloCluster& cluster) const ;
  void setTowerCollection(const CaloTowerCollection* towercollection);
  bool hasActiveHcal( const reco::SuperCluster & sc ) const ;
  bool hasActiveHcal( const std::vector<CaloTowerDetId> & towers ) const ;


 private:
  const CaloTowerConstituentsMap * towerMap_;
  HoeMode mode_;
  const CaloTowerCollection * towerCollection_;
  unsigned int NMaxClusters_;
  const HcalChannelQuality * hcalQuality_;
};

bool ClusterGreaterThan(const reco::CaloClusterPtr& c1, const reco::CaloClusterPtr& c2) ;

#endif
