#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/VectorHolder.h"

namespace {
  namespace {
    edm::Wrapper<std::vector<reco::GenParticleCandidate> > w1;
    reco::CompositeRefCandidateT<reco::GenParticleRefVector> v1;
    edm::Wrapper<reco::GenParticleCollection> w2;
    edm::Wrapper<reco::GenParticleMatch> w3;
    reco::GenParticleRef r1;
    reco::GenParticleRefProd rp1;
    edm::Wrapper<reco::GenParticleRefVector> wrv1;
    edm::reftobase::Holder<reco::Candidate, reco::GenParticleRef> hcg1;
    edm::reftobase::RefHolder<reco::GenParticleRef> hcg2;
    edm::reftobase::VectorHolder<reco::Candidate, reco::GenParticleRefVector> hcg3;
    edm::reftobase::RefVectorHolder<reco::GenParticleRefVector> hcg4;
  }
}
