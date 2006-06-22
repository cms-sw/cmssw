// $Id: RecoCandidate.cc,v 1.8 2006/06/05 14:12:04 llista Exp $
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
