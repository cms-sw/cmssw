#include "RecoMuon/MuonIdentification/interface/MuonCosmicsId.h"
#include "DataFormats/TrackReco/interface/Track.h"

bool directionAlongMomentum(const reco::Track& track){
  // check is done in 2D
  return (track.innerPosition().x()-track.vx())*track.px() +
    (track.innerPosition().y()-track.vy())*track.py() > 0;
}

reco::TrackRef 
muonid::findOppositeTrack(const edm::Handle<reco::TrackCollection>& tracks, 
			const reco::Track& muonTrack,
			double angleMatch,
			double momentumMatch)
{
  for (unsigned int i=0; i<tracks->size(); ++i){
    // When both tracks are reconstructed as outside going, sign is -1
    // otherwise it's +1. There is also a crazy case of both are outside
    // going, then sign is -1 as well.
    int match_sign = directionAlongMomentum(muonTrack)==directionAlongMomentum(tracks->at(i)) ? -1 : +1; 
    double sprod = muonTrack.px()*tracks->at(i).px() + 
      muonTrack.py()*tracks->at(i).py() + muonTrack.pz()*tracks->at(i).pz();
    if ( acos( match_sign*(sprod/tracks->at(i).p()/muonTrack.p()) ) < angleMatch &&
	 fabs(tracks->at(i).pt()-muonTrack.pt())/muonTrack.pt() < momentumMatch )
      return reco::TrackRef(tracks,i);
  }
  return reco::TrackRef();

}
