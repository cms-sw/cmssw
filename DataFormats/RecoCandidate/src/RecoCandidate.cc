#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

RecoCandidate::~RecoCandidate() { }

RecoCandidate * RecoCandidate::clone() const {
   throw cms::Exception("LogicError", "reco::RecoCandidate is abstract, so it's clone() method can't be implemented.\n");
}

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

TrackBaseRef RecoCandidate::bestTrackRef() const {
  TrackRef muRef = combinedMuon();
  if( muRef.isNonnull() ) return TrackBaseRef(muRef);
  TrackRef trkRef = track();
  if ( trkRef.isNonnull() ) return TrackBaseRef(trkRef);
  GsfTrackRef gsfTrkRef = gsfTrack();
  if ( gsfTrkRef.isNonnull() ) return TrackBaseRef(gsfTrkRef);
  TrackRef staRef = standAloneMuon(); 
  if ( staRef.isNonnull() ) return TrackBaseRef(staRef); 
  return TrackBaseRef();
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

float RecoCandidate::dzError() const 
{
  const Track * tr=bestTrack(); 
  if(tr!=nullptr) 
    return tr->dzError(); 
  else 
    return 0; 
}
float RecoCandidate::dxyError() const 
{
  const Track * tr=bestTrack(); 
  if(tr!=nullptr) 
    return tr->dxyError(); 
  else 
    return 0; 
}
