#ifndef RecoCandidate_RecoPFClusterRefCandidate_h
#define RecoCandidate_RecoPFClusterRefCandidate_h

#include "DataFormats/Candidate/interface/LeafRefCandidateT.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

namespace reco {


  typedef LeafRefCandidateT<PFClusterRef> RecoPFClusterRefCandidateBase;
  

  class RecoPFClusterRefCandidate : public  RecoPFClusterRefCandidateBase {
  public:
    RecoPFClusterRefCandidate() : LeafRefCandidateT<PFClusterRef>() {}
    RecoPFClusterRefCandidate(PFClusterRef ref, float m) : LeafRefCandidateT<PFClusterRef>( ref, m) {}
    
    ~RecoPFClusterRefCandidate() {};

    reco::PFClusterRef const & pfCluster() const {
      return ref_;
    }
  };
}

#endif
