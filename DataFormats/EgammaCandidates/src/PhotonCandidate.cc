// $Id: PhotonCandidate.cc,v 1.1 2006/04/21 06:28:47 llista Exp $
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
  SuperClusterRef s1 = superCluster(), s2 = dstc->superCluster();
  if ( ! s1.isNull() && ! s2.isNull() && s1 == s2 ) return true;
  return false;
}
