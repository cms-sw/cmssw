// $Id: RecoElectronCandidate.cc,v 1.4 2006/02/21 10:37:36 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoElectronCandidate.h"
#include "DataFormats/EGammaReco/interface/Electron.h"

using namespace reco;

RecoElectronCandidate::~RecoElectronCandidate() { }

RecoElectronCandidate * RecoElectronCandidate::clone() const { 
  return new RecoElectronCandidate( * this ); 
}

ElectronRef RecoElectronCandidate::electron() const {
  return electron_;
}

TrackRef RecoElectronCandidate::track() const {
  return electron_->track();
}

SuperClusterRef RecoElectronCandidate::superCluster() const {
  return electron_->superCluster();
}

