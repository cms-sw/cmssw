// $Id: Photon.cc,v 1.6 2007/03/24 23:13:30 futyand Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 

using namespace reco;

Photon::~Photon() { }

Photon * Photon::clone() const { 
  return new Photon( * this ); 
}

reco::SuperClusterRef Photon::superCluster() const {
  return superCluster_;
}

bool Photon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}

void Photon::setVertex(const Point & vertex) {
  math::XYZVector direction = caloPosition() - vertex;
  double energy = this->energy();
  math::XYZVector momentum = direction.unit() * energy;
  p4_.SetXYZT(momentum.x(), momentum.y(), momentum.z(), energy );
  vertex_ = vertex;
}

math::XYZPoint Photon::caloPosition() const {
  if (r9_>0.93) {
    return unconvPosition_;
  } else {
    return superCluster()->position();
  }
}
