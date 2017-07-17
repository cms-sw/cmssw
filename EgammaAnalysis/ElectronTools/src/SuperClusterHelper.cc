#include "EgammaAnalysis/ElectronTools/interface/SuperClusterHelper.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"




SuperClusterHelper::SuperClusterHelper(const reco::GsfElectron * electron, const EcalRecHitCollection * rechits, const CaloTopology * topo, const CaloGeometry* geom) {
  theElectron_  = electron;
  rechits_ = rechits ;
  seedCluster_ = & (*(electron->superCluster()->seed()));
  theSuperCluster_ = &(*electron->superCluster());
  
  eSubClusters_ = 0.;
  // Store subclusters
  reco::CaloCluster_iterator itscl = theSuperCluster_->clustersBegin();
  reco::CaloCluster_iterator itsclE = theSuperCluster_->clustersEnd();
  for(; itscl < itsclE ; ++itscl) {
    if((*itscl)==electron->superCluster()->seed()) continue; // skip seed cluster
    theBasicClusters_.push_back(&(**itscl));  
    eSubClusters_ += (*itscl)->energy();
  }
  // sort subclusters
  sort(theBasicClusters_.begin(), theBasicClusters_.end(), SuperClusterHelper::sortClusters);
  // Add seed cluster at the beginning
  theBasicClusters_.insert(theBasicClusters_.begin(), seedCluster_);

  nBasicClusters_ = theBasicClusters_.size();

  // Store ES clusters
  eESClusters_ = 0. ;
  itscl = theSuperCluster_->preshowerClustersBegin();
  itsclE = theSuperCluster_->preshowerClustersEnd();
  for( ; itscl < itsclE ; ++ itscl) {
    theESClusters_.push_back(&(**itscl)); 
    eESClusters_ += (*itscl)->energy(); 
  }
  // sort ES clusters
  sort(theESClusters_.begin(), theESClusters_.end(), SuperClusterHelper::sortClusters);

  nESClusters_ = theESClusters_.size();

  topology_ = topo;
  geometry_ = geom;
  barrel_ = electron->isEB();
  covComputed_ = false;
  localCoordinatesComputed_ = false;
}

SuperClusterHelper::SuperClusterHelper(const pat::Electron * electron, const EcalRecHitCollection * rechits, const CaloTopology * topo, const CaloGeometry * geom) {
  theElectron_  = (const reco::GsfElectron*)electron;
  rechits_ = rechits ;
//  for(unsigned ir=0; ir<rechits_->size();++ir) {
//    std::cout << "RecHit " << (*rechits_)[ir].id().rawId() << " " << (*rechits_)[ir] << std::endl;
//  }
  // Get the embedded objects
  theSuperCluster_ = &(*electron->superCluster());
  seedCluster_ = & (*(electron->seed()));
  const std::vector<reco::CaloCluster>& basicClusters(electron->basicClusters());
  nBasicClusters_ = basicClusters.size();
  eSubClusters_ = 0. ;
  // Store subclusters
  for ( unsigned ib = 0; ib < nBasicClusters_ ; ++ib) {
    if(fabs((basicClusters[ib].energy()-seedCluster_->energy())/seedCluster_->energy())<1.e-5 &&
       fabs((basicClusters[ib].eta()-seedCluster_->eta())/seedCluster_->eta())<1.e-5 &&
       fabs((basicClusters[ib].phi()-seedCluster_->phi())/seedCluster_->phi())<1.e-5 
       ) 
        continue; // skip seed cluster
    theBasicClusters_.push_back(&basicClusters[ib]); 
    eSubClusters_ += basicClusters[ib].energy();
  }
  // sort subclusters
  sort(theBasicClusters_.begin(), theBasicClusters_.end(), SuperClusterHelper::sortClusters);
  // Add seed cluster at the beginning
  theBasicClusters_.insert(theBasicClusters_.begin(), seedCluster_);
	  
  // Store ES clusters
  const std::vector<reco::CaloCluster>& esClusters(electron->preshowerClusters());
  nESClusters_ = esClusters.size();
  eESClusters_ = 0. ;
  for (unsigned ib = 0 ; ib < nESClusters_ ; ++ ib) {
    theESClusters_.push_back(&esClusters[ib]);
    eESClusters_ += esClusters[ib].energy();
  }
  // sort ES clusters
  sort(theESClusters_.begin(), theESClusters_.end(), SuperClusterHelper::sortClusters);

//  std::vector< std::pair<DetId, float> >::const_iterator it=seedCluster_->hitsAndFractions().begin();
//  std::vector< std::pair<DetId, float> >::const_iterator itend=seedCluster_->hitsAndFractions().end();
//  for( ; it!=itend ; ++it) {
//    DetId id=it->first;
//    std::cout << " Basic cluster " << id.rawId() << std::endl;
//  }
  topology_ = topo;
  geometry_ = geom;
  barrel_ = electron->isEB();
  covComputed_ = false;
  localCoordinatesComputed_ = false; 
}

void SuperClusterHelper::computeLocalCovariances() {
  if (!covComputed_) {
    vCov_ = EcalClusterTools::localCovariances(*seedCluster_, rechits_, topology_, 4.7);
    covComputed_ = true;
    
    spp_ = 0;
    if (!isnan(vCov_[2])) spp_ = sqrt (vCov_[2]);
    
    if (theElectron_->sigmaIetaIeta()*spp_ > 0) {
      sep_ = vCov_[1] / (theElectron_->sigmaIetaIeta() * spp_);
    } else if (vCov_[1] > 0) {
      sep_ = 1.0;
    } else {
      sep_ = -1.0;
    }
  }
}

float SuperClusterHelper::spp() {
  computeLocalCovariances();
  return spp_;
}

float SuperClusterHelper::sep() {
  computeLocalCovariances();
  return sep_;
}

void SuperClusterHelper::localCoordinates() {
  if (localCoordinatesComputed_) return;

  if (barrel_) {
    local_.localCoordsEB(*seedCluster_, *geometry_, etaCrySeed_ , phiCrySeed_ ,ietaSeed_ , iphiSeed_ , thetaTilt_ , phiTilt_);
  } else {
    local_.localCoordsEE(*seedCluster_, *geometry_, etaCrySeed_ , phiCrySeed_ ,ietaSeed_ , iphiSeed_ , thetaTilt_ , phiTilt_);
  }
    localCoordinatesComputed_ = true;
}


float SuperClusterHelper::subClusterEnergy(unsigned i) const {
  return (nBasicClusters_ > i) ? theBasicClusters_[i]->energy() : 0.; 
}

float SuperClusterHelper::subClusterEta(unsigned i) const {
  return (nBasicClusters_ > i) ? theBasicClusters_[i]->eta() : 999.; 
}

float SuperClusterHelper::subClusterPhi(unsigned i) const {
  return (nBasicClusters_ > i) ? theBasicClusters_[i]->phi() : 999.; 
}

float SuperClusterHelper::subClusterEmax(unsigned i) const {
  return (nBasicClusters_ > i) ? EcalClusterTools::eMax(*theBasicClusters_[i], rechits_) : 0.; 
}

float SuperClusterHelper::subClusterE3x3(unsigned i) const {
  return (nBasicClusters_ > i) ? EcalClusterTools::e3x3(*theBasicClusters_[i], rechits_, topology_) : 0.; 
}

float SuperClusterHelper::esClusterEnergy(unsigned i) const {
  return (nESClusters_ > i) ? theESClusters_[i]->energy() : 0. ;
}

float SuperClusterHelper::esClusterEta(unsigned i) const {
  return (nESClusters_ > i) ? theESClusters_[i]->eta() : 999. ;
}

float SuperClusterHelper::esClusterPhi(unsigned i) const {
  return (nESClusters_ > i) ? theESClusters_[i]->phi() : 999. ;
}
