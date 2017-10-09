#ifndef RecoCandidate_RecoPFClusterRefCandidate_h
#define RecoCandidate_RecoPFClusterRefCandidate_h

#include "DataFormats/Candidate/interface/LeafRefCandidateT.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

namespace reco {


  typedef LeafRefCandidateT  RecoPFClusterRefCandidateBase;
  

  class RecoPFClusterRefCandidate : public  LeafRefCandidateT {
  public:
    RecoPFClusterRefCandidate() : LeafRefCandidateT() {}
    RecoPFClusterRefCandidate(PFClusterRef ref, float m) : LeafRefCandidateT( ref, m) {}
    
    ~RecoPFClusterRefCandidate() {}

    RecoPFClusterRefCandidate * clone() const { return new RecoPFClusterRefCandidate(*this);}

    reco::PFClusterRef pfCluster() const {
      return getRef<reco::PFClusterRef>();
    }
  };
}

#endif
