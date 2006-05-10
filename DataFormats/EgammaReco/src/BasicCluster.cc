#include "DataFormats/EgammaReco/interface/BasicCluster.h"

using namespace reco;

BasicCluster::BasicCluster( double energy, const Point& position, double chi2, const std::vector<DetId> usedHits, AlgoId algoID) :
   EcalCluster(energy,position), chi2_(chi2), usedHits_(usedHits)
{
  algoId_ = algoID;
}



/**
BasicCluster::BasicCluster( const std::vector<EcalRecHitData>& recHits,
                           int superClusterId, const Point & position ) :
    superClusterId_( superClusterId ), 
    recHits_( recHits ) {

  double energy=0.;
  std::vector<EcalRecHitData>::const_iterator it;
  for (it = recHits.begin(); it != recHits.end(); it++) {
    energy += it->energy() * it->fraction();
    chi2_ += it->energy() * it->chi2();
    usedHits_.push_back(it->detId());
  }
  chi2_/=energy;
}
**/

bool BasicCluster::operator<(const reco::BasicCluster &otherCluster) const
{
  if(otherCluster.energy() > energy()) 
    return false;
  else
    return true;
}

bool BasicCluster::operator==(const BasicCluster& rhs) const  
{
  
  float Ediff = fabs(rhs.energy() - energy());
  if (Ediff < 0.00000001) return true;
  else return false;

}
