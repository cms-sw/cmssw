#ifndef RecoAlgos_PFClusterToRefCandidate_h
#define RecoAlgos_PFClusterToRefCandidate_h
#include "CommonTools/RecoAlgos/src/MassiveCandidateConverter.h"
#include "CommonTools/RecoAlgos/src/CandidateProducer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace converter {

  struct PFClusterToRefCandidate : public MassiveCandidateConverter {
    typedef reco::PFCluster value_type;
    typedef reco::PFClusterCollection Components;
    typedef reco::RecoPFClusterRefCandidate Candidate;
    PFClusterToRefCandidate(const edm::ParameterSet & cfg) : 
      MassiveCandidateConverter(cfg) {
    }
    void convert(reco::PFClusterRef pfclusterRef, reco::RecoPFClusterRefCandidate & c) const {
      c = reco::RecoPFClusterRefCandidate( pfclusterRef, sqrt(massSqr_) );
    }  
  };

  namespace helper {
    template<>
    struct CandConverter<reco::PFCluster> { 
      typedef PFClusterToRefCandidate type;
    };
  }

}

#endif
