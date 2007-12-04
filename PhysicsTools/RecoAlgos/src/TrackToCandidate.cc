#include "PhysicsTools/RecoAlgos/src/TrackToCandidate.h"

using namespace reco;
using namespace converter;

void TrackToCandidate::convert( size_t idx, const edm::Handle<reco::TrackCollection> & tracks, 
				RecoChargedCandidate & c ) const {
  TrackRef trkRef( tracks, idx );
  const Track & trk = * trkRef;
  c.setCharge( trk.charge() );
  c.setVertex( trk.vertex() );
  Track::Vector p = trk.momentum();
  double t = sqrt( massSqr_ + p.mag2() );
  c.setP4( Candidate::LorentzVector( p.x(), p.y(), p.z(), t ) );
  c.setTrack( trkRef );
  c.setPdgId( particle_.pdgId() );
}
