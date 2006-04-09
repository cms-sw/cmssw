#include "DataFormats/EgammaReco/interface/BasicCluster.h"

using namespace reco;

BasicCluster::BasicCluster( const std::vector<EcalRecHitData>& recHits,
			    int superClusterId, const Point & position ) :
  position_( position ), superClusterId_( superClusterId_ ), 
  recHits_( recHits ) {
  energy_=0.;
  std::vector<EcalRecHitData>::const_iterator it;
  for (it = recHits.begin(); it != recHits.end(); it++) {
    energy_ += it->energy() * it->fraction();
    chi2_ += it->energy() * it->chi2();
  }
  chi2_/=energy_;
}

bool BasicCluster::operator<(const reco::BasicCluster &otherCluster) const
{
  if(otherCluster.energy() > energy()) 
    return false;
  else
    return true;
}
