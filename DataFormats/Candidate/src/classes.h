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
#include "DataFormats/Candidate/interface/CandMatchMapMany.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

namespace {
  namespace {
    std::vector<reco::Candidate *> v1;
    reco::CandidateCollection o1;
    edm::Wrapper<reco::CandidateCollection> w1;
    std::vector<reco::Particle> v2;
    edm::Wrapper<std::vector<reco::Particle> > w2;
    reco::CandidateRef r1;
    reco::CandidateBaseRef r2;
    edm::reftobase::IndirectHolder<reco::Candidate> rbih1;
    edm::reftobase::RefHolder<reco::CandidateRef> rh1;
    edm::Wrapper<reco::CandidateRefVector> wrv1;
    edm::Wrapper<reco::CandidateBaseRefVector> wrv2;
    reco::CandidateRefProd rp1;
    std::vector<edm::RefToBase<reco::Candidate> > vrb1;
    edm::reftobase::Holder<reco::Candidate, reco::CandidateRef> rhcr1;
    edm::reftobase::VectorHolder<reco::Candidate, reco::CandidateRefVector> rhcr2;
    edm::Wrapper<reco::CandFloatAssociations> wav1;
    edm::Wrapper<reco::CandDoubleAssociations> wav2;
    edm::Wrapper<reco::CandIntAssociations> wav3;
    edm::Wrapper<reco::CandUIntAssociations> wav4;
    edm::helpers::KeyVal<reco::CandidateRef,reco::CandidateRef> kv1;
    reco::CandMatchMap cmm1;
    reco::CandMatchMap::const_iterator cmm1it;
    edm::Wrapper<reco::CandMatchMap> wcmm1;
    edm::helpers::KeyVal<reco::CandidateRefProd, reco::CandidateRefProd> kv2;
    std::map<const reco::Candidate *, const reco::Candidate *> m1;
    std::vector<const reco::Candidate *> vc1;
    reco::CandMatchMapMany cmm2;
    edm::Wrapper<reco::CandMatchMapMany> wcmm2;
    edm::Wrapper<std::vector<reco::CandidateBaseRef> > wvrb1;
  }
}
