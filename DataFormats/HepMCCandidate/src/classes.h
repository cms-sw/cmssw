#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"

namespace {
  namespace {
    edm::Wrapper<std::vector<reco::GenParticleCandidate> > w1;
    reco::CompositeRefCandidateT<reco::GenParticleRefVector> v1;
    edm::Wrapper<reco::GenParticleCollection> w2;
    reco::GenParticleRef r1;
    reco::GenParticleRefVector rv1;
    edm::reftobase::Holder<reco::Candidate, reco::GenParticleRef> hcg1;
    edm::reftobase::RefHolder<reco::GenParticleRef> hcg2;
  }
}
