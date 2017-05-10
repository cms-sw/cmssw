#ifndef RecoCandidate_RecoChargedRefCandidate_h
#define RecoCandidate_RecoChargedRefCandidate_h

#include "DataFormats/Candidate/interface/LeafRefCandidateT.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {


  typedef LeafRefCandidateT  RecoChargedRefCandidateBase;
  

  class RecoChargedRefCandidate : public  RecoChargedRefCandidateBase {
  public:
    RecoChargedRefCandidate() {}
    RecoChargedRefCandidate(TrackRef ref, float m) : LeafRefCandidateT( ref, m) {}
    
    ~RecoChargedRefCandidate() {}

    RecoChargedRefCandidate * clone() const { return new RecoChargedRefCandidate(*this);}

    reco::TrackRef track() const {
      return getRef<reco::TrackRef>();
    }
  };
}

#endif
