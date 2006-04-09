// $Id: SuperCluster.cc,v 1.3 2006/02/17 08:34:43 llista Exp $
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h"
using namespace reco;

SuperCluster::SuperCluster( const Vector & m, const Point & p, double uE ) :
  momentum_( m ), position_( p ), uncorrectedEnergy_( uE ) {
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

double SuperCluster::disc1() const { 
  return pi0Discriminator_->disc1(); 
}

double SuperCluster::disc2() const { 
  return pi0Discriminator_->disc2(); 
}

double SuperCluster::disc3() const { 
  return pi0Discriminator_->disc3(); 
}
