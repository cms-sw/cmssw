#include "RecoMuon/MuonIdentification/interface/MuonCosmicsId.h"
#include "DataFormats/TrackReco/interface/Track.h"

bool directionAlongMomentum(const reco::Track& track){
  // check is done in 2D
  return (track.innerPosition().x()-track.vx())*track.px() +
    (track.innerPosition().y()-track.vy())*track.py() > 0;
}

// returns angle and dPt/Pt
std::pair<double, double> 
muonid::matchTracks(const reco::Track& ref, const reco::Track& probe)
{
  std::pair<double,double> result(0,0);
  // When both tracks are reconstructed as outside going, sign is -1
  // otherwise it's +1. There is also a crazy case of both are outside
  // going, then sign is -1 as well.
  int match_sign = directionAlongMomentum(ref)==directionAlongMomentum(probe) ? -1 : +1; 
  double sprod = ref.px()*probe.px() + ref.py()*probe.py() + ref.pz()*probe.pz();
  double argCos = match_sign*(sprod/ref.p()/probe.p());
  if (argCos < -1.0) argCos = -1.0;
  if (argCos > 1.0) argCos = 1.0;
  result.first = acos( argCos );
  result.second = fabs(probe.pt()-ref.pt())/sqrt(ref.pt()*probe.pt()); //SK: take a geom-mean pt
  return result;
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
    const std::pair<double,double>& match = matchTracks(muonTrack,tracks->at(i));
    if ( match.first < angleMatch && match.second < momentumMatch )
      return reco::TrackRef(tracks,i);
  }
  return reco::TrackRef();

}
