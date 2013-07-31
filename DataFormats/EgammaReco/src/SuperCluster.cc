// $Id: SuperCluster.cc,v 1.17 2011/02/17 22:42:03 argiro Exp $
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include <algorithm>
using namespace reco;

namespace {
  typedef reco::SuperCluster::EEtoPSAssociationInternal::value_type sorttype;
  bool compareKey(const sorttype& a, const sorttype& b) {
    return a.first < b.first;
  }
  const std::vector<size_t> noassociations;
}

SuperCluster::SuperCluster( double energy, const math::XYZPoint& position ) :
  CaloCluster( energy, position ), preshowerEnergy_(0), rawEnergy_(0), phiWidth_(0), etaWidth_(0), preshowerEnergy1_(0), preshowerEnergy2_(0) {
}



SuperCluster::SuperCluster( double energy, const math::XYZPoint& position,
                            const CaloClusterPtr & seed,
                            const CaloClusterPtrVector& clusters,
			    double Epreshower, double phiWidth, double etaWidth, double Epreshower1, double Epreshower2) :
  CaloCluster(energy,position), rawEnergy_(0)
{
  phiWidth_ = phiWidth;
  etaWidth_ = etaWidth;
  seed_ = seed;
  preshowerEnergy_ = Epreshower;
  preshowerEnergy1_ = Epreshower1;
  preshowerEnergy2_ = Epreshower2;

  // set references to constituent basic clusters and update list of rechits
  for(CaloClusterPtrVector::const_iterator bcit  = clusters.begin();
                                            bcit != clusters.end();
                                          ++bcit) {
    clusters_.push_back( (*bcit) );

    // updated list of used hits
    const std::vector< std::pair<DetId, float> > & v1 = (*bcit)->hitsAndFractions();
    for( std::vector< std::pair<DetId, float> >::const_iterator diIt = v1.begin();
                                            diIt != v1.end();
                                           ++diIt ) {
      hitsAndFractions_.push_back( (*diIt) );
    } // loop over rechits
  } // loop over basic clusters
  
  computeRawEnergy();
}



SuperCluster::SuperCluster( double energy, const math::XYZPoint& position,
                            const CaloClusterPtr & seed,
                            const CaloClusterPtrVector& clusters,
                            const CaloClusterPtrVector& preshowerClusters,
			    double Epreshower, double phiWidth, double etaWidth, double Epreshower1, double Epreshower2) :
  CaloCluster(energy,position), rawEnergy_(-1.)
{
  phiWidth_ = phiWidth;
  etaWidth_ = etaWidth;
  seed_ = seed;
  preshowerEnergy_ = Epreshower;
  preshowerEnergy1_ = Epreshower1;
  preshowerEnergy2_ = Epreshower2;

  // set references to constituent basic clusters and update list of rechits
  for(CaloClusterPtrVector::const_iterator bcit  = clusters.begin();
                                            bcit != clusters.end();
                                          ++bcit) {
    clusters_.push_back( (*bcit) );

    // updated list of used hits
    const std::vector< std::pair<DetId, float> > & v1 = (*bcit)->hitsAndFractions();
    for( std::vector< std::pair<DetId, float> >::const_iterator diIt = v1.begin();
                                            diIt != v1.end();
                                           ++diIt ) {
      hitsAndFractions_.push_back( (*diIt) );
    } // loop over rechits
  } // loop over basic clusters

  // set references to preshower clusters
  for(CaloClusterPtrVector::const_iterator pcit  = preshowerClusters.begin();
                                            pcit != preshowerClusters.end();
                                          ++pcit) {
    preshowerClusters_.push_back( (*pcit) );
  }
  computeRawEnergy();
}

void SuperCluster::addPreshowerCluster( const CaloClusterPtr & ee,
					const CaloClusterPtr & ps ) {
  EEtoPSAssociationInternal::iterator begin(ee2ps.begin()), end(ee2ps.end());
  const EEtoPSAssociationInternal::value_type 
    key(ee.key(),std::vector<size_t>());
  EEtoPSAssociationInternal::value_type to_insert(key);
  size_t idx = std::distance(preshowerClusters_.begin(),
			     preshowerClusters_.end());
  EEtoPSAssociationInternal::iterator entry = std::lower_bound(begin,
							       end,
							       key,
							       compareKey);
  if( entry != end && entry->first == key.first ) {
    entry->second.push_back(idx);         
  } else { 
    to_insert.second.push_back(idx);    
    ee2ps.insert(entry,to_insert);
  }
  preshowerClusters_.push_back(ps);
}

const std::vector<size_t>& 
SuperCluster::preshowerClustersAssociated(const CaloClusterPtr& i) const { 
  const EEtoPSAssociationInternal::value_type 
    key(i.key(),std::vector<size_t>());
  EEtoPSAssociationInternal::const_iterator entry = 
    std::lower_bound(ee2ps.cbegin(),ee2ps.cend(),key,compareKey);
  if( entry == ee2ps.end() || entry->first != i.key() ) {
    return noassociations;
  }
  return entry->second; 
}

void SuperCluster::computeRawEnergy() {

  rawEnergy_ = 0.;
  for(CaloClusterPtrVector::const_iterator bcItr = clustersBegin(); 
      bcItr != clustersEnd(); bcItr++){
      rawEnergy_ += (*bcItr)->energy();
  }
}
