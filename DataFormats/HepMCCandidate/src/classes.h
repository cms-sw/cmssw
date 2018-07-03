#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/VectorHolder.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/HepMCCandidate/interface/PdfInfo.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistoryEvent.h"
#include <vector>

namespace DataFormats_HepMCCandidate {
  struct dictionary {
    reco::CompositeRefCandidateT<reco::GenParticleRefVector> v1;
    edm::Wrapper<reco::GenParticleCollection> w2;
    edm::Wrapper<reco::GenParticleMatch> w3;
    reco::GenParticleRef r1;
    reco::GenParticleRefProd rp1;
    edm::Wrapper<reco::GenParticleRefVector> wrv1;
    edm::Wrapper<std::vector<std::vector<reco::GenParticleRef> > > wvvg1;
    edm::reftobase::Holder<reco::Candidate, reco::GenParticleRef> hcg1;
    edm::reftobase::RefHolder<reco::GenParticleRef> hcg2;
    edm::reftobase::VectorHolder<reco::Candidate, reco::GenParticleRefVector> hcg3;
    edm::reftobase::RefVectorHolder<reco::GenParticleRefVector> hcg4;
    reco::PdfInfo p1;
    edm::Wrapper<reco::PdfInfo> wp1;
    reco::FlavorHistory fh1;
    std::vector<reco::FlavorHistory> vfh1;
    edm::Wrapper<std::vector<reco::FlavorHistory> > wvfh1;
    edm::ValueMap<reco::FlavorHistory> vmfh1;
    edm::Wrapper<edm::ValueMap<reco::FlavorHistory> > wvmfh1;
    reco::FlavorHistoryEvent fhe1;
    std::vector<reco::FlavorHistoryEvent> vfhe1;
    edm::Wrapper<reco::FlavorHistoryEvent > wfhe1;
    edm::Wrapper<std::vector<reco::FlavorHistoryEvent> > wvfhe1;
    edm::ValueMap<reco::FlavorHistoryEvent> vmfhe1;
    edm::Wrapper<edm::ValueMap<reco::FlavorHistoryEvent> > wvmfhe1;
    std::vector<reco::GenParticleRef>	v_gpr;
    edm::RefVectorIterator<std::vector<reco::GenParticle>,reco::GenParticle,edm::refhelper::FindUsingAdvance<std::vector<reco::GenParticle>,reco::GenParticle> > rvigp;
    edm::ValueMap<edm::Ref<std::vector<reco::GenParticle>,reco::GenParticle,edm::refhelper::FindUsingAdvance<std::vector<reco::GenParticle>,reco::GenParticle> > > vmgr;
    edm::Wrapper<edm::ValueMap<edm::Ref<std::vector<reco::GenParticle>,reco::GenParticle,edm::refhelper::FindUsingAdvance<std::vector<reco::GenParticle>,reco::GenParticle> > > > wvmgr;
    edm::Ptr<reco::GenParticle> gpptr;
    edm::FwdPtr<reco::GenParticle> gpfp;
    std::vector<edm::FwdPtr<reco::GenParticle>> vgpfp;
    edm::Wrapper<std::vector<edm::FwdPtr<reco::GenParticle>>> wvgpfp;    
  };
}

