// $Id: RecoPhotonCandidate.cc,v 1.2 2006/04/10 08:05:32 llista Exp $
#include "DataFormats/EgammaCandidates/interface/PhotonCandidate.h"

using namespace reco;

PhotonCandidate::~PhotonCandidate() { }

PhotonCandidate * PhotonCandidate::clone() const { 
  return new PhotonCandidate( * this ); 
}

reco::SuperClusterRef PhotonCandidate::superCluster() const {
  return superCluster_;
}
