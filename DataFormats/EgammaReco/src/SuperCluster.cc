// $Id: SuperCluster.cc,v 1.11 2008/02/11 13:10:24 kkaadze Exp $
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
using namespace reco;

SuperCluster::SuperCluster( double energy, const math::XYZPoint& position ) :
  CaloCluster( energy, position ), rawEnergy_(-1.) {
}

SuperCluster::SuperCluster( double energy, const math::XYZPoint& position,
                            const BasicClusterRef & seed,
                            const BasicClusterRefVector& clusters,
			    double Epreshower, double phiWidth, double etaWidth) :
  CaloCluster(energy,position), rawEnergy_(-1.)
{
  phiWidth_ = phiWidth;
  etaWidth_ = etaWidth;
  seed_ = seed;
  preshowerEnergy_ = Epreshower;

  // set references to constituent basic clusters and update list of rechits
  for(BasicClusterRefVector::const_iterator bcit  = clusters.begin();
                                            bcit != clusters.end();
                                          ++bcit) {
    clusters_.push_back( (*bcit) );

    // updated list of used hits
    const std::vector<DetId> & v1 = (*bcit)->getHitsByDetId();
    for( std::vector<DetId>::const_iterator diIt = v1.begin();
                                            diIt != v1.end();
                                           ++diIt ) {
      usedHits_.push_back( (*diIt) );
    } // loop over rechits
  } // loop over basic clusters

}

double SuperCluster::rawEnergy() const
{
  if (rawEnergy_<0) {
    rawEnergy_ = 0.;
    reco::basicCluster_iterator bcItr;
    for(bcItr = clustersBegin(); bcItr != clustersEnd(); bcItr++)
      {
	rawEnergy_ += (*bcItr)->energy();
      }
  }
  return rawEnergy_;
}
