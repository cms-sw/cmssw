#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    edm::Wrapper<std::vector<reco::GenParticleCandidate> > w1;
    reco::CompositeRefCandidateT<reco::GenParticleRefVector> v1;
    edm::Wrapper<std::vector<reco::GenParticle> > w2;
    reco::GenParticleRef r1;
    reco::GenParticleRefVector rv1;
  }
}
