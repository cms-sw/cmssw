// $Id: SuperCluster.cc,v 1.2 2006/04/12 15:19:15 rahatlou Exp $
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h"
using namespace reco;

SuperCluster::SuperCluster( double energy, const math::XYZPoint& position ) :
  EcalCluster(energy,position) {
}

SuperCluster::SuperCluster( double energy, const math::XYZPoint& position,
                            const BasicClusterRef & seed,
                            const BasicClusterRefVector& clusters ) :
  EcalCluster(energy,position),
  seed_(seed_),
  clusters_(clusters) {
}




std::vector<DetId>
SuperCluster::getHitsByDetId() const {
  std::vector<DetId> usedHits;
  return usedHits;
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

double SuperCluster::hadOverEcal() const{
  return shape_->hadOverEcal();
}

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
