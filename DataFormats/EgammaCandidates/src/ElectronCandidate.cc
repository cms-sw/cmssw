// $Id: ElectronCandidate.cc,v 1.2 2006/04/10 08:05:32 llista Exp $
#include "DataFormats/EgammaCandidates/interface/ElectronCandidate.h"

using namespace reco;

ElectronCandidate::~ElectronCandidate() { }

ElectronCandidate * ElectronCandidate::clone() const { 
  return new ElectronCandidate( * this ); 
}

TrackRef ElectronCandidate::track() const {
  return track_;
}

SuperClusterRef ElectronCandidate::superCluster() const {
  return superCluster_;
}

