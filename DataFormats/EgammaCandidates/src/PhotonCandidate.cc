// $Id: PhotonCandidate.cc,v 1.2 2006/04/26 07:56:20 llista Exp $
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
  const RecoCandidate * dstc = dynamic_cast<const RecoCandidate *>( & c );
  if ( dstc == 0 ) return false;
  if ( checkOverlap( superCluster(), dstc->superCluster() ) ) return true;
  return false;
}
