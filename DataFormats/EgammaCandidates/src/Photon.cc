// $Id: Photon.cc,v 1.2 2006/06/16 15:01:16 llista Exp $
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
  math::XYZVector direction = superCluster()->position() - vertex;
  math::XYZVector momentum = direction.unit() * superCluster()->energy();
  p4_.SetXYZT(momentum.x(), momentum.y(), momentum.z(), superCluster()->energy() );
  vertex_ = vertex;
}
