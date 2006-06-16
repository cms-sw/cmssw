// $Id: PhotonCandidate.cc,v 1.5 2006/05/31 12:57:40 llista Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"

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
