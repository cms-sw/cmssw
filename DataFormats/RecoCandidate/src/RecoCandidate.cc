// $Id: RecoCandidate.cc,v 1.10 2006/07/27 07:13:42 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace reco;

RecoCandidate::~RecoCandidate() { }

TrackRef RecoCandidate::track() const {
  return TrackRef();
}

TrackRef RecoCandidate::track( size_t ) const {
  return TrackRef();
}

size_t RecoCandidate::numberOfTracks() const {
  return 0;
}

GsfTrackRef RecoCandidate::gsfTrack() const {
  return GsfTrackRef();
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
