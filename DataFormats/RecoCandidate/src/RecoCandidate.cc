// $Id: RecoCandidate.cc,v 1.7 2006/05/02 10:28:01 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace reco;

RecoCandidate::~RecoCandidate() { }

TrackRef RecoCandidate::track() const {
  return TrackRef();
}

TrackRef RecoCandidate::standAloneMuon() const {
  return TrackRef();
}

TrackRef RecoCandidate::combinedMuon() const {
  return TrackRef();
}

SuperClusterRef RecoCandidate::superCluster() const {
  return SuperClusterRef();
}

CaloTowerRef RecoCandidate::caloTower() const {
  return CaloTowerRef();
}

CaloJetRef RecoCandidate::caloJet() const {
  return CaloJetRef();
}
