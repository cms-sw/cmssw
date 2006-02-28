#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<reco::Candidate *> v1;
    reco::CandidateCollection o1;
    edm::Wrapper<reco::CandidateCollection> w1;
    reco::CandidateRef r1;
    reco::CandidateRefs rv1;
    reco::CandidatesRef rp1;
  }
}
