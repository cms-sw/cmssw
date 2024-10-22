#ifndef RecoAlgos_PFClusterToRefCandidate_h
#define RecoAlgos_PFClusterToRefCandidate_h
#include "CommonTools/RecoAlgos/interface/MassiveCandidateConverter.h"
#include "CommonTools/RecoAlgos/interface/CandidateProducer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace converter {

  struct PFClusterToRefCandidate : public MassiveCandidateConverter {
    typedef reco::PFCluster value_type;
    typedef reco::PFClusterCollection Components;
    typedef reco::RecoPFClusterRefCandidate Candidate;
    PFClusterToRefCandidate(const edm::ParameterSet& cfg, edm::ConsumesCollector iC)
        : MassiveCandidateConverter(cfg, iC) {}
    void convert(reco::PFClusterRef pfclusterRef, reco::RecoPFClusterRefCandidate& c) const {
      c = reco::RecoPFClusterRefCandidate(pfclusterRef, sqrt(massSqr_));
    }
  };

  namespace helper {
    template <>
    struct CandConverter<reco::PFCluster> {
      typedef PFClusterToRefCandidate type;
    };
  }  // namespace helper

}  // namespace converter

#endif
