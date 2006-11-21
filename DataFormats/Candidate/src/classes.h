#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

namespace {
  namespace {
    std::vector<reco::Candidate *> v1;
    reco::CandidateCollection o1;
    edm::Wrapper<reco::CandidateCollection> w1;
    std::vector<reco::Particle> v2;
    edm::Wrapper<std::vector<reco::Particle> > w2;
    reco::CandidateRef r1;
    reco::CandidateBaseRef r2;
    reco::CandidateRefVector rv1;
    reco::CandidateRefProd rp1;
    std::vector<edm::RefToBase<reco::Candidate> > vrb1;
    edm::reftobase::Holder<reco::Candidate, reco::CandidateRef> rhcr1;
    edm::AssociationVector<reco::CandidateCollection, std::vector<double> > av1;
    edm::Wrapper<edm::AssociationVector<reco::CandidateCollection, std::vector<double> > > wav1;
    edm::AssociationVector<reco::CandidateCollection, std::vector<float> > av2;
    edm::Wrapper<edm::AssociationVector<reco::CandidateCollection, std::vector<float> > > wav2;
    edm::AssociationVector<reco::CandidateCollection, std::vector<int> > av3;
    edm::Wrapper<edm::AssociationVector<reco::CandidateCollection, std::vector<int> > > wav3;
    edm::helpers::KeyVal<reco::CandidateRef,reco::CandidateRef> kv1;
    reco::CandMatchMap cmm1;
    edm::Wrapper<reco::CandMatchMap> wcmm1;
  }
}
