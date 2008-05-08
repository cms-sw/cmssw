#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h"
#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/CandMatchMapMany.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/VectorHolder.h"

namespace {
  namespace {
    std::vector<reco::Candidate *> v1;
    reco::CandidateCollection o1;
    edm::Wrapper<reco::CandidateCollection> w1;
    std::vector<reco::Particle> v2;
    edm::Wrapper<std::vector<reco::Particle> > w2;
    edm::Wrapper<std::vector<reco::LeafCandidate> > w2bis;
    reco::CandidateRef r1;
    reco::CandidatePtr ptr1;
    reco::CandidateBaseRef r2;
    reco::CompositeCandidateRefVector r3;
    std::vector<reco::CandidateBaseRef> rv2;
    edm::reftobase::IndirectHolder<reco::Candidate> rbih1;
    edm::reftobase::RefHolder<reco::CandidateRef> rh1;
    edm::reftobase::IndirectVectorHolder<reco::Candidate> rbih2;
    edm::reftobase::RefVectorHolder<reco::CandidateRefVector> rh2;
    edm::reftobase::Holder<reco::Candidate, reco::CandidateRef> rhcr1;
    edm::reftobase::VectorHolder<reco::Candidate, reco::CandidateRefVector> rhcr2;
    edm::Wrapper<reco::CandidateRefVector> wrv1;
    edm::Wrapper<reco::CandidateBaseRefVector> wrv2;
    edm::Wrapper<reco::CompositeCandidateRefVector> wrv3;
    reco::CandidateRefProd rp1;
    reco::CandidateBaseRefProd rp2;
    std::vector<edm::RefToBase<reco::Candidate> > vrb1;
    std::vector<edm::Ptr<reco::Candidate> > vrb2;
    edm::Wrapper<reco::CandFloatAssociations> wav1;
    edm::Wrapper<reco::CandDoubleAssociations> wav2;
    edm::Wrapper<reco::CandIntAssociations> wav3;
    edm::Wrapper<reco::CandUIntAssociations> wav4;
    edm::Wrapper<reco::CandViewFloatAssociations> wav5;
    edm::Wrapper<reco::CandViewDoubleAssociations> wav6;
    edm::Wrapper<reco::CandViewIntAssociations> wav7;
    edm::Wrapper<reco::CandViewUIntAssociations> wav8;
    edm::helpers::KeyVal<reco::CandidateRef,reco::CandidateRef> kv1;
    reco::CandMatchMap cmm1;
    reco::CandMatchMap::const_iterator cmm1it;
    edm::Wrapper<reco::CandMatchMap> wcmm1;
    edm::helpers::KeyVal<reco::CandidateRefProd, reco::CandidateRefProd> kv2;
    reco::CandViewMatchMap cmm2;
    reco::CandViewMatchMap::const_iterator cmm2it;
    edm::Wrapper<reco::CandViewMatchMap> wcmm2;
    edm::helpers::KeyVal<reco::CandidateBaseRefProd, reco::CandidateBaseRefProd> kv3;
    std::map<const reco::Candidate *, const reco::Candidate *> m1;
    std::vector<const reco::Candidate *> vc1;
    reco::CandMatchMapMany cmm3;
    reco::CandMatchMapMany::const_iterator cmm3it;
    edm::Wrapper<reco::CandMatchMapMany> wcmm3;
    edm::Wrapper<std::vector<reco::CandidateBaseRef> > wvrb1;
    edm::Wrapper<edm::Association<reco::CandidateCollection> > wacc1;
    edm::Wrapper<reco::CompositeCandidateCollection> wcc1;
    edm::Wrapper<reco::CandRefValueMap> wcrvm1;
    edm::reftobase::Holder<reco::Candidate, reco::CompositeCandidateRef> hcc1;
    edm::reftobase::RefHolder<reco::CompositeCandidateRef> hcc2;
    edm::reftobase::VectorHolder<reco::Candidate, reco::CompositeCandidateRefVector> hcc3;
    edm::reftobase::RefVectorHolder<reco::CompositeCandidateRefVector> hcc4;
    edm::Wrapper<reco::VertexCompositeCandidateCollection> wcc2;
    edm::reftobase::Holder<reco::Candidate, reco::VertexCompositeCandidateRef> hcc5;
    edm::reftobase::RefHolder<reco::VertexCompositeCandidateRef> hcc6;
    edm::reftobase::VectorHolder<reco::Candidate, reco::VertexCompositeCandidateRefVector> hcc7;
    edm::reftobase::RefVectorHolder<reco::VertexCompositeCandidateRefVector> hcc8;
    edm::Wrapper<reco::NamedCompositeCandidateCollection> wcc3;
    edm::reftobase::Holder<reco::Candidate, reco::NamedCompositeCandidateRef> hcc9;
    edm::reftobase::RefHolder<reco::NamedCompositeCandidateRef> hcc10;
    edm::reftobase::VectorHolder<reco::Candidate, reco::NamedCompositeCandidateRefVector> hcc11;
    edm::reftobase::RefVectorHolder<reco::NamedCompositeCandidateRefVector> hcc12;
  }
}
