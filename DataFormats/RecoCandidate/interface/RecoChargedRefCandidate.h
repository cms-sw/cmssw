#ifndef RecoCandidate_RecoChargedRefCandidate_h
#define RecoCandidate_RecoChargedRefCandidate_h

#include "DataFormats/Candidate/interface/LeafRefCandidateT.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {


  typedef LeafRefCandidateT  RecoChargedRefCandidateBase;
  

  class RecoChargedRefCandidate : public  LeafRefCandidateT {
  public:
    RecoChargedRefCandidate() {}
    RecoChargedRefCandidate(TrackRef ref, float m) : LeafRefCandidateT( ref, m) {}
    
    ~RecoChargedRefCandidate() {}

    RecoChargedRefCandidate * clone() const { return new RecoChargedRefCandidate(*this);}

    reco::TrackRef track() const {
      return getRef<reco::TrackRef>();
    }
    // return a pointer to the best track, if available.
    // otherwise, return a null pointer
    virtual const reco::Track * bestTrack() const {
      if ( track().isNonnull() && track().isAvailable() )
        return &(*track());
      else
        return nullptr;
    }

    /// uncertainty on dz 
    virtual float dzError() const { const Track * tr=bestTrack(); if(tr!=nullptr) return tr->dzError(); else return 0; }
    /// uncertainty on dxy
    virtual float dxyError() const { const Track * tr=bestTrack(); if(tr!=nullptr) return tr->dxyError(); else return 0; }
 
  };
}

#endif
