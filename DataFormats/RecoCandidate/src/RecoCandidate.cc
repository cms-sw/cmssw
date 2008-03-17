// $Id: RecoCandidate.cc,v 1.12 2007/04/02 13:35:17 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

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

const Track * RecoCandidate::bestTrack() const {
  TrackRef muRef = combinedMuon();
  if( muRef.isNonnull() ) 
    return muRef.get();
  TrackRef trkRef = track();
  if ( trkRef.isNonnull() ) 
    return trkRef.get();
  GsfTrackRef gsfTrkRef = gsfTrack();
  if ( gsfTrkRef.isNonnull() )
    return gsfTrkRef.get();
  TrackRef staRef = standAloneMuon(); 
  if ( staRef.isNonnull() ) 
    return staRef.get(); 
  return 0;
}

RecoCandidate::TrackType RecoCandidate::bestTrackType() const {
  if( combinedMuon().isNonnull() ) 
    return recoTrackType;
  if ( track().isNonnull() ) 
    return recoTrackType;
  if ( gsfTrack().isNonnull() ) 
    return gsfTrackType;
  return noTrackType;
}
