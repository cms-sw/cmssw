#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

using namespace reco;

IsolatedPixelTrackCandidate::~IsolatedPixelTrackCandidate() { }

IsolatedPixelTrackCandidate * IsolatedPixelTrackCandidate::clone() const { 
  return new IsolatedPixelTrackCandidate( * this ); 
}

TrackRef IsolatedPixelTrackCandidate::track() const {
  return track_;
}

l1extra::L1JetParticleRef IsolatedPixelTrackCandidate::l1tau() const {
  return l1tauJet_;
}

bool IsolatedPixelTrackCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&  checkOverlap( track(), o->track() ) );
}
