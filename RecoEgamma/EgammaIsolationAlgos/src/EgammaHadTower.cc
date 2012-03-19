#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include <algorithm>
#include <iostream>

EgammaHadTower::EgammaHadTower(const edm::EventSetup &es,HoeMode mode):mode_(mode) {
  edm::ESHandle<CaloTowerConstituentsMap> ctmaph;
  es.get<IdealGeometryRecord>().get(ctmaph);
  towerMap_ = &(*ctmaph);
  NMaxClusters_ = 4;
}

CaloTowerDetId  EgammaHadTower::towerOf(const reco::CaloCluster& cluster) const {
  DetId detid = cluster.seed();
  if(detid.det() != DetId::Ecal) {
    // Basic clusters of hybrid super-cluster do not have the seed set; take the first DetId instead 
    // Should be checked . The single Tower Mode should be favoured until fixed
    detid = cluster.hitsAndFractions()[0].first;
    if(detid.det() != DetId::Ecal) {
      CaloTowerDetId tower;
      return tower;
    }
  }
  CaloTowerDetId id(towerMap_->towerOf(detid));
  return id;
}

std::vector<CaloTowerDetId>  EgammaHadTower::towersOf(const reco::SuperCluster& sc) const {
  std::vector<CaloTowerDetId> towers;
  std::vector<reco::CaloClusterPtr>  orderedClusters;

  // in this mode, check only the tower behind the seed
  if ( mode_ == SingleTower ) {
    towers.push_back(towerOf(*sc.seed()));
  }

  // in this mode check the towers behind each basic cluster
  if ( mode_ == TowersBehindCluster ) {
    // Loop on the basic clusters
    reco::CaloCluster_iterator it = sc.clustersBegin();
    reco::CaloCluster_iterator itend = sc.clustersEnd();

    for ( ; it !=itend; ++it) {
      orderedClusters.push_back(*it);
    }
    std::sort(orderedClusters.begin(),orderedClusters.end(),ClusterGreaterThan);
    unsigned nclusters=orderedClusters.size();
    for ( unsigned iclus =0 ; iclus <nclusters && iclus < NMaxClusters_; ++iclus) {
      // Get the tower
      CaloTowerDetId id = towerOf(*(orderedClusters[iclus]));
      std::vector<CaloTowerDetId>::const_iterator itcheck=find(towers.begin(),towers.end(),id);
      if( itcheck == towers.end() ) {
	towers.push_back(id);
      }
    }
  }
//  if(towers.size() > 4) {
//    std::cout << " NTOWERS " << towers.size() << " ";
//    for(unsigned i=0; i<towers.size() ; ++i) {
//      std::cout << towers[i] << " ";
//    }
//    std::cout <<  std::endl;
//    for ( unsigned iclus=0 ; iclus < orderedClusters.size(); ++iclus) {
//      // Get the tower
//      CaloTowerDetId id = towerOf(*(orderedClusters[iclus]));
//      std::cout << " Pos " << orderedClusters[iclus]->position() << " " << orderedClusters[iclus]->energy() << " " << id ;
//    }
//    std::cout << std::endl;
//  }
  return towers;
}

double EgammaHadTower::getDepth1HcalESum(const std::vector<CaloTowerDetId> & towers) const {
  double esum=0.;
  CaloTowerCollection::const_iterator trItr = towerCollection_->begin();
  CaloTowerCollection::const_iterator trItrEnd = towerCollection_->end();
  for( ;  trItr != trItrEnd ; ++trItr){
    std::vector<CaloTowerDetId>::const_iterator itcheck = find(towers.begin(), towers.end(), trItr->id());
    if( itcheck != towers.end() ) {
      esum += trItr->ietaAbs()<18 || trItr->ietaAbs()>29 ? trItr->hadEnergy() : trItr->hadEnergyHeInnerLayer() ;
    }
  }
  return esum;
}

double EgammaHadTower::getDepth2HcalESum(const std::vector<CaloTowerDetId> & towers) const {
  double esum=0.;
  CaloTowerCollection::const_iterator trItr = towerCollection_->begin();
  CaloTowerCollection::const_iterator trItrEnd = towerCollection_->end();
  for( ;  trItr != trItrEnd ; ++trItr){
    std::vector<CaloTowerDetId>::const_iterator itcheck = find(towers.begin(), towers.end(), trItr->id());
    if( itcheck != towers.end() ) {
      esum += trItr->hadEnergyHeOuterLayer();
    }
  }
  return esum;
}

double EgammaHadTower::getDepth1HcalESum( const reco::SuperCluster& sc ) const {
  return getDepth1HcalESum(towersOf(sc)) ;
}

double EgammaHadTower::getDepth2HcalESum( const reco::SuperCluster& sc ) const {
  return getDepth2HcalESum(towersOf(sc)) ;
}

void EgammaHadTower::setTowerCollection(const CaloTowerCollection* towerCollection) {
  towerCollection_ = towerCollection;
}

bool ClusterGreaterThan(const reco::CaloClusterPtr& c1, const reco::CaloClusterPtr& c2)  {
  return (*c1 > *c2);
}
