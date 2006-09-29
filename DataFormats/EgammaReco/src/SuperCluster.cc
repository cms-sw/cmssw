// $Id: SuperCluster.cc,v 1.5 2006/05/23 16:28:06 askew Exp $
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h"
using namespace reco;

SuperCluster::SuperCluster( double energy, const math::XYZPoint& position ) :
  EcalCluster( energy, position ) {
}

SuperCluster::SuperCluster( double energy, const math::XYZPoint& position,
                            const BasicClusterRef & seed,
                            const BasicClusterRefVector& clusters ) :
  EcalCluster(energy,position)
{

  seed_ = seed;

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

std::vector<DetId>
SuperCluster::getHitsByDetId() const {
  return usedHits_;
}



double SuperCluster::eMax() const{
  return shape_->eMax();
}

double SuperCluster::e2x2() const{
  return shape_->e2x2();
}

double SuperCluster::e3x3() const{
  return shape_->e3x3();
}

double SuperCluster::e5x5() const{
  return shape_->e5x5();
}

double SuperCluster::covEtaEta() const{
  return shape_->covEtaEta();
}

double SuperCluster::covEtaPhi() const{
  return shape_->covEtaPhi();
}

double SuperCluster::covPhiPhi() const{
  return shape_->covPhiPhi();
}

// double SuperCluster::hadOverEcal() const{
//   return shape_->hadOverEcal();
// }

/**
double SuperCluster::disc1() const { 
  return pi0Discriminator_->disc1(); 
}

double SuperCluster::disc2() const { 
  return pi0Discriminator_->disc2(); 
}

double SuperCluster::disc3() const { 
  return pi0Discriminator_->disc3(); 
}
**/
