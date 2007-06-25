// $Id: Photon.cc,v 1.5 2007/02/19 21:37:16 futyand Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

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
  math::XYZVector momentum = direction.unit() * superCluster()->energy();
  p4_.SetXYZT(momentum.x(), momentum.y(), momentum.z(), superCluster()->energy() );
  vertex_ = vertex;
}

math::XYZPoint Photon::caloPosition() const {
  if (r9_>0.93) {
    return unconvPosition_;
  } else {
    return superCluster()->position();
  }
}
