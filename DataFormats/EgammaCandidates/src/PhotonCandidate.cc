// $Id: PhotonCandidate.cc,v 1.3 2006/05/02 10:19:02 llista Exp $
#include "DataFormats/EgammaCandidates/interface/PhotonCandidate.h"

using namespace reco;

PhotonCandidate::~PhotonCandidate() { }

PhotonCandidate * PhotonCandidate::clone() const { 
  return new PhotonCandidate( * this ); 
}

reco::SuperClusterRef PhotonCandidate::superCluster() const {
  return superCluster_;
}

bool PhotonCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && ! 
	   ( checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}
