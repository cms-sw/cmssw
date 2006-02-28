// $Id: RecoPhotonCandidate.cc,v 1.4 2006/02/21 10:37:36 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoPhotonCandidate.h"
#include "DataFormats/EGammaReco/interface/Photon.h"

using namespace reco;

RecoPhotonCandidate::~RecoPhotonCandidate() { }

RecoPhotonCandidate * RecoPhotonCandidate::clone() const { 
  return new RecoPhotonCandidate( * this ); 
}

PhotonRef RecoPhotonCandidate::photon() const {
  return photon_;
}

SuperClusterRef RecoPhotonCandidate::superCluster() const {
  return photon_->superCluster();
}
