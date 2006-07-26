#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace {
  namespace {
    std::vector<reco::Candidate *> v1;
    reco::CandidateCollection o1;
    edm::Wrapper<reco::CandidateCollection> w1;
    reco::CandidateRef r1;
    reco::CandidateRefVector rv1;
    reco::CandidateRefProd rp1;

    edm::RefToBase<reco::Candidate> * rb1;
    std::vector<edm::RefToBase<reco::Candidate> > vrb1;

    edm::reftobase::Holder<reco::Candidate, reco::CandidateRef> rhcr1;
  }
}
