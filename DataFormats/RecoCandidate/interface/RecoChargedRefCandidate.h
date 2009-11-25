#ifndef RecoCandidate_RecoChargedRefCandidate_h
#define RecoCandidate_RecoChargedRefCandidate_h

#include "DataFormats/Candidate/interface/LeafRefCandidateT.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {


  typedef LeafRefCandidateT<TrackRef> RecoChargedRefCandidateBase;
  

  class RecoChargedRefCandidate : public  RecoChargedRefCandidateBase {
  public:
    RecoChargedRefCandidate() : LeafRefCandidateT<TrackRef>() {}
    RecoChargedRefCandidate(TrackRef ref, float m) : LeafRefCandidateT<TrackRef>( ref, m) {}
    
    ~RecoChargedRefCandidate() {};

    reco::TrackRef const & track() const {
      return ref_;
    }
  };
}

#endif
